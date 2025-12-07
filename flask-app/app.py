from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import numpy as np
from datetime import datetime
import pickle
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
import threading
import concurrent.futures
import requests
import json
import time
from functools import lru_cache
import hashlib
import shutil
import subprocess
import fcntl  # 파일 잠금용
import os

from services.captioning import generate_caption
from services.text_processing import preprocess_text, expand_search_query

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB upload limit

# ========== FAISS & 임베딩 모델 설정 ==========
FAISS_STORAGE_DIR = os.getenv("FAISS_STORAGE_DIR", "/app/faiss-data")
FAISS_INDEX_PATH = os.path.join(FAISS_STORAGE_DIR, 'faiss_index.idx')
FAISS_MAPPING_PATH = os.path.join(FAISS_STORAGE_DIR, 'id_mapping.pkl')
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# FAISS 인덱스 타입 설정 (HNSW: 대량 데이터 검색 최적화, Flat: 소량 데이터)
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "HNSW")  # "HNSW" or "Flat"
HNSW_M = int(os.getenv("HNSW_M", "32"))  # HNSW 파라미터: 연결 수 (16-64, 높을수록 정확하지만 느림)
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))  # HNSW 빌드 시 탐색 범위
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "128"))  # HNSW 검색 시 탐색 범위 (높을수록 정확하지만 느림)

# 검색 성능 개선: 유사도 임계값 설정
# IndexFlatIP은 내적 값 (코사인 유사도 * 벡터 크기), 정규화된 벡터는 0~1 범위
# BGE-M3는 정규화된 임베딩을 사용하므로 코사인 유사도는 대략 0.3~0.95 범위
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # 최소 유사도 임계값 (0.3 = 30%)
MIN_RESULTS_TO_RETURN = int(os.getenv("MIN_RESULTS_TO_RETURN", "3"))  # 최소 반환 결과 수

# ========== 전역 변수 ==========
faiss_index = None
id_mapping = {}  # FAISS 인덱스 번호 -> MySQL item_id 매핑
embedding_model: Optional[SentenceTransformer] = None
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
_faiss_initialized = False
_model_loaded = False
_faiss_lock = threading.Lock()  # FAISS 인덱스 접근 동기화
_pending_save_count = 0  # 저장 대기 중인 벡터 개수
_save_batch_size = 10  # N개마다 저장 (개별 API용)

def check_and_free_disk_space(min_free_gb: float = 5.0):
    """디스크 공간 확인 및 필요시 정리"""
    try:
        # 디스크 사용량 확인
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_percent = (stat.used / stat.total) * 100
        
        print(f"💾 디스크 상태: 전체 {total_gb:.2f}GB, 사용 {used_percent:.1f}%, 여유 {free_gb:.2f}GB")
        
        if free_gb < min_free_gb:
            print(f"⚠️ 디스크 공간 부족 ({free_gb:.2f}GB < {min_free_gb}GB). 정리 시작...")
            
            # 임시 파일 정리
            for tmp_dir in ["/tmp", "/var/tmp"]:
                if os.path.exists(tmp_dir):
                    try:
                        for item in os.listdir(tmp_dir):
                            item_path = os.path.join(tmp_dir, item)
                            try:
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                            except Exception as e:
                                pass  # 권한 문제 등 무시
                    except Exception as e:
                        pass
            
            # pip 캐시 정리
            pip_cache = os.path.expanduser("~/.cache/pip")
            if os.path.exists(pip_cache):
                try:
                    shutil.rmtree(pip_cache)
                except:
                    pass
            
            # Python 캐시 정리
            for root, dirs, files in os.walk("/usr/local/lib/python3.10"):
                for d in dirs:
                    if d == "__pycache__":
                        try:
                            shutil.rmtree(os.path.join(root, d))
                        except:
                            pass
            
            # 다시 확인
            stat = shutil.disk_usage("/")
            free_gb_after = stat.free / (1024**3)
            print(f"✅ 정리 완료: 여유 공간 {free_gb_after:.2f}GB")
            
            if free_gb_after < min_free_gb:
                print(f"❌ 경고: 여전히 디스크 공간이 부족합니다 ({free_gb_after:.2f}GB)")
                return False
        
        return True
    except Exception as e:
        print(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True  # 실패해도 계속 진행

def _try_load_faiss_file() -> tuple[bool, Optional[object], Optional[dict]]:
    """FAISS 파일 로드 시도 (손상된 파일 자동 처리)"""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_MAPPING_PATH):
        return False, None, None
    
    try:
        # 파일 크기 확인 (손상된 파일 사전 감지)
        index_size = os.path.getsize(FAISS_INDEX_PATH)
        if index_size == 0:
            print(f"⚠️ FAISS 인덱스 파일 크기가 0입니다. 손상된 파일로 간주합니다.")
            return False, None, None
        
        # 파일 읽기 전에 파일이 완전히 쓰여졌는지 확인
        # 파일 크기가 일정 시간 동안 변하지 않으면 완전히 쓰여진 것으로 간주
        import time
        prev_size = index_size
        for _ in range(5):  # 최대 5번 확인 (0.1초 간격)
            time.sleep(0.1)
            current_size = os.path.getsize(FAISS_INDEX_PATH)
            if current_size != prev_size:
                # 파일이 아직 쓰여지고 있음
                prev_size = current_size
            else:
                break
        
        # 파일 읽기 시도 (명시적으로 예외 처리)
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        except RuntimeError as e:
            # FAISS 읽기 에러 (파일 손상 등)
            error_msg = str(e)
            if "read error" in error_msg or "ret == (size)" in error_msg:
                print(f"❌ FAISS 파일 읽기 에러 감지: {error_msg}")
                print(f"   파일 크기: {os.path.getsize(FAISS_INDEX_PATH)} bytes")
                return False, None, None
            else:
                # 다른 종류의 RuntimeError는 재발생
                raise
        
        # 매핑 파일 읽기
        try:
            with open(FAISS_MAPPING_PATH, 'rb') as f:
                id_mapping = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, IOError) as e:
            print(f"❌ FAISS 매핑 파일 읽기 실패: {type(e).__name__}: {e}")
            return False, None, None
        
        # 검증: 인덱스와 매핑의 일관성 확인
        if faiss_index.ntotal != len(id_mapping):
            print(f"⚠️ FAISS 인덱스와 매핑 불일치: 인덱스={faiss_index.ntotal}, 매핑={len(id_mapping)}")
            return False, None, None
        
        return True, faiss_index, id_mapping
    except (RuntimeError, IOError, OSError, Exception) as e:
        # 손상된 파일 감지 (모든 예외 타입 명시적으로 처리)
        print(f"❌ FAISS 파일 손상 감지: {type(e).__name__}: {e}")
        import traceback
        print(f"   상세 에러:\n{traceback.format_exc()}")
        return False, None, None

def initialize_faiss():
    """FAISS 인덱스 초기화 또는 로드 (한 번만 실행)
    
    인덱스 타입:
    - IndexFlatIP: 정확하지만 느림 (소량 데이터용, < 10만개)
    - IndexHNSWFlat: 빠르고 정확함 (대량 데이터용, > 10만개)
    """
    global faiss_index, id_mapping, _faiss_initialized
    
    if _faiss_initialized:
        return
    
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
    
    # 파일 잠금 경로
    lock_file_path = os.path.join(FAISS_STORAGE_DIR, '.faiss_lock')
    
    # 파일 잠금을 먼저 획득하여 다른 워커와의 충돌 방지
    # 여러 워커가 동시에 시작될 때를 대비하여 타임아웃 설정
    lock_file = None
    lock_acquired = False
    max_retries = 10
    retry_delay = 0.5  # 0.5초
    
    for retry in range(max_retries):
        try:
            lock_file = open(lock_file_path, 'w')
            # 논블로킹 잠금 시도 (LOCK_NB)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
            break
        except (IOError, OSError) as e:
            # 다른 워커가 잠금을 보유 중이면 대기
            if lock_file:
                lock_file.close()
            if retry < max_retries - 1:
                print(f"⏳ 파일 잠금 대기 중... (시도 {retry + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                # 최대 재시도 횟수 초과 시 블로킹 잠금 시도
                print(f"⚠️ 논블로킹 잠금 실패, 블로킹 잠금 시도...")
                lock_file = open(lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                lock_acquired = True
                break
    
    if not lock_acquired:
        raise RuntimeError("FAISS 파일 잠금 획득 실패")
    
    try:
        # 파일 로드 시도 (손상된 파일 자동 처리)
        success, loaded_index, loaded_mapping = _try_load_faiss_file()
        
        if success and loaded_index is not None and loaded_mapping is not None:
            # 파일 로드 성공
            faiss_index = loaded_index
            id_mapping = loaded_mapping
            index_type_name = type(faiss_index).__name__
            index_size = os.path.getsize(FAISS_INDEX_PATH)
            print(f"✅ FAISS 인덱스 로드: {faiss_index.ntotal}개 벡터 (타입: {index_type_name}, 크기: {index_size / 1024 / 1024:.2f}MB)")
            
            # HNSW 인덱스인 경우 ef_search 설정
            if hasattr(faiss_index, 'hnsw'):
                faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
                print(f"   HNSW 파라미터 설정: ef_search={HNSW_EF_SEARCH}")
        else:
            # 파일이 없거나 손상된 경우
            if os.path.exists(FAISS_INDEX_PATH) or os.path.exists(FAISS_MAPPING_PATH):
                # 손상된 파일인 경우 백업 후 삭제
                print(f"🔄 손상된 파일을 백업하고 새 인덱스를 생성합니다...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if os.path.exists(FAISS_INDEX_PATH):
                    backup_path = f"{FAISS_INDEX_PATH}.corrupted_{timestamp}"
                    try:
                        shutil.move(FAISS_INDEX_PATH, backup_path)
                        print(f"   백업: {backup_path}")
                    except Exception as e:
                        print(f"   백업 실패, 파일 삭제: {e}")
                        try:
                            os.remove(FAISS_INDEX_PATH)
                        except:
                            pass
                
                if os.path.exists(FAISS_MAPPING_PATH):
                    backup_path = f"{FAISS_MAPPING_PATH}.corrupted_{timestamp}"
                    try:
                        shutil.move(FAISS_MAPPING_PATH, backup_path)
                        print(f"   백업: {backup_path}")
                    except Exception as e:
                        print(f"   백업 실패, 파일 삭제: {e}")
                        try:
                            os.remove(FAISS_MAPPING_PATH)
                        except:
                            pass
                
                # 손상된 파일이 있었으므로 새 인덱스 생성 (메모리상에서만, 저장은 나중에)
                if FAISS_INDEX_TYPE.upper() == "HNSW":
                    faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
                    faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
                    faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
                    print(f"✅ HNSW FAISS 인덱스 생성 (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, ef_search={HNSW_EF_SEARCH})")
                else:
                    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                    print("✅ Flat FAISS 인덱스 생성 (정확한 검색)")
                
                id_mapping = {}
            else:
                # 파일이 없는 경우: CI/CD 시 초기화 방지
                # 빈 인덱스를 메모리에만 생성 (디스크에 저장하지 않음)
                # CSV 임포트 API 호출 시 데이터가 추가되면 그때 저장됨
                print(f"⚠️ FAISS 인덱스 파일이 없습니다. 빈 인덱스로 시작합니다 (CSV 임포트 API 호출 시 데이터 추가됨).")
                if FAISS_INDEX_TYPE.upper() == "HNSW":
                    faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
                    faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
                    faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
                else:
                    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
                id_mapping = {}
                print(f"   인덱스 타입: {FAISS_INDEX_TYPE}, 벡터 개수: 0 (CSV 임포트로 데이터 추가 필요)")
    except Exception as e:
        # 예상치 못한 예외 발생 시에도 새 인덱스 생성 (메모리상에서만)
        print(f"❌ FAISS 초기화 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 손상된 파일 백업 시도
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{FAISS_INDEX_PATH}.corrupted_{timestamp}"
                shutil.move(FAISS_INDEX_PATH, backup_path)
                print(f"   손상된 파일 백업: {backup_path}")
        except:
            pass
        
        # 새 인덱스 생성 (메모리상에서만, 디스크 저장은 나중에)
        if FAISS_INDEX_TYPE.upper() == "HNSW":
            faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
            faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        else:
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        id_mapping = {}
        print(f"✅ 새 FAISS 인덱스 생성 완료 (예외 복구, 메모리상에서만)")
    finally:
        # 잠금 해제
        if lock_file and lock_acquired:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except:
                pass
            lock_file.close()
    
    _faiss_initialized = True

def save_faiss():
    """FAISS 스냅샷 저장 (파일 잠금으로 멀티 워커 충돌 방지)"""
    global faiss_index, id_mapping
    
    if faiss_index is None:
        print("⚠️ FAISS 인덱스가 없어 저장을 건너뜁니다.")
        return
    
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
    
    # 파일 잠금 경로
    lock_file_path = os.path.join(FAISS_STORAGE_DIR, '.faiss_lock')
    
    # 파일 잠금을 사용하여 멀티 워커 동시 접근 방지
    try:
        with open(lock_file_path, 'w') as lock_file:
            # 배타적 잠금 획득 (다른 워커는 대기)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            try:
                # 저장 전 디스크 공간 확인 (최소 1GB 필요)
                stat = shutil.disk_usage(FAISS_STORAGE_DIR)
                free_gb = stat.free / (1024**3)
                if free_gb < 1.0:
                    print(f"❌ 디스크 공간 부족으로 저장 실패 (여유: {free_gb:.2f}GB)")
                    raise RuntimeError(f"디스크 공간 부족: {free_gb:.2f}GB < 1.0GB")
                
                # 원자적 쓰기: 임시 파일에 쓰고 성공 후 원본으로 이동
                temp_index_path = f"{FAISS_INDEX_PATH}.tmp"
                temp_mapping_path = f"{FAISS_MAPPING_PATH}.tmp"
                
                # 임시 파일에 저장
                faiss.write_index(faiss_index, temp_index_path)
                
                with open(temp_mapping_path, 'wb') as f:
                    pickle.dump(id_mapping, f)
                
                # 파일 크기 확인 (손상 방지)
                temp_index_size = os.path.getsize(temp_index_path)
                if temp_index_size == 0:
                    raise RuntimeError("임시 인덱스 파일 크기가 0입니다")
                
                # 기존 파일이 있으면 백업
                if os.path.exists(FAISS_INDEX_PATH):
                    backup_path = f"{FAISS_INDEX_PATH}.backup"
                    try:
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                        shutil.copy2(FAISS_INDEX_PATH, backup_path)
                    except:
                        pass  # 백업 실패해도 계속 진행
                
                # 원자적 이동 (성공 시에만 원본 파일 교체)
                shutil.move(temp_index_path, FAISS_INDEX_PATH)
                shutil.move(temp_mapping_path, FAISS_MAPPING_PATH)
                
                # 최종 확인 (파일 존재 및 크기 검증)
                if not os.path.exists(FAISS_INDEX_PATH):
                    raise RuntimeError("인덱스 파일이 저장되지 않았습니다")
                
                final_size = os.path.getsize(FAISS_INDEX_PATH)
                if final_size == 0:
                    raise RuntimeError("저장된 인덱스 파일 크기가 0입니다")
                
                if final_size != temp_index_size:
                    raise RuntimeError(f"파일 크기 불일치: 예상 {temp_index_size}, 실제 {final_size}")
                
                print(f"💾 FAISS 저장 완료: {faiss_index.ntotal}개 벡터 ({temp_index_size / 1024 / 1024:.2f}MB)")
                
            finally:
                # 잠금 해제 (자동으로 해제됨)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
    except BlockingIOError:
        # 다른 워커가 사용 중이면 저장 건너뜀
        print("⚠️ 다른 워커가 파일을 사용 중이어 저장을 건너뜁니다.")
        return
    except Exception as e:
        # 실패 시 임시 파일 정리
        for temp_path in [f"{FAISS_INDEX_PATH}.tmp", f"{FAISS_MAPPING_PATH}.tmp"]:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        print(f"❌ FAISS 저장 실패: {e}")
        raise

# ========== AI 팀이 구현할 함수들 (현재는 더미) ==========


def load_embedding_model() -> SentenceTransformer:
    """SentenceTransformer 모델을 1회 로드"""
    global embedding_model, _model_loaded
    if embedding_model is None or not _model_loaded:
        # 모델 로드 전 디스크 공간 확인
        check_and_free_disk_space(min_free_gb=2.0)
        print(f"📦 BGE 모델 로드: {EMBEDDING_MODEL_NAME} (device={embedding_device})")
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=embedding_device,
            trust_remote_code=True,
        )
        embedding_model.max_seq_length = 512
        _model_loaded = True
        print(f"✅ BGE 모델 로드 완료")
    return embedding_model

def describe_image_with_llava(image_bytes):
    """
    이미지에서 자연어 설명 생성 (Qwen 기반).
    원본 캡션을 반환하고, 전처리는 나중에 통합적으로 수행.
    """
    try:
        caption = generate_caption(image_bytes)
        # 원본 캡션 반환 (전처리는 나중에 통합적으로 수행)
        return caption.strip() if caption else ""
    except Exception as exc:
        print(f"⚠️ 이미지 캡셔닝 실패: {exc}")
        return ""

# 임베딩 캐시 (자주 사용되는 텍스트의 임베딩 캐싱하여 리소스 절약)
_embedding_cache = {}
_embedding_cache_lock = threading.Lock()
EMBEDDING_CACHE_SIZE = 1000  # 최대 캐시 크기


def _get_text_hash(text: str) -> str:
    """텍스트의 해시값 생성 (캐시 키용)"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def create_embedding_vector(text: str, use_cache: bool = True):
    """
    BGE-M3 모델을 사용하여 텍스트를 임베딩 벡터로 변환
    
    리소스 절약: 자주 사용되는 텍스트는 캐싱하여 재사용
    
    Args:
        text (str): 임베딩할 텍스트 (이미지 묘사 + 사용자 입력 설명)
        use_cache (bool): 캐시 사용 여부 (기본값: True)
        
    Returns:
        numpy.ndarray: shape (EMBEDDING_DIMENSION,) 임베딩 벡터
    """
    if not text or not text.strip():
        raise ValueError("임베딩할 텍스트가 비어 있습니다.")

    # 캐시 확인 (리소스 절약)
    if use_cache:
        text_hash = _get_text_hash(text)
        with _embedding_cache_lock:
            if text_hash in _embedding_cache:
                return _embedding_cache[text_hash].copy()

    model = load_embedding_model()
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=1,  # 배치 크기 명시
    )[0].astype("float32")

    if embedding.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"임베딩 차원 불일치: 기대값={EMBEDDING_DIMENSION}, 실제값={embedding.shape[0]}"
        )

    # 캐시 저장 (리소스 절약)
    if use_cache:
        with _embedding_cache_lock:
            # 캐시 크기 제한 (LRU 방식으로 오래된 것 제거)
            if len(_embedding_cache) >= EMBEDDING_CACHE_SIZE:
                # 가장 오래된 항목 제거 (간단한 방식: 랜덤 제거)
                oldest_key = next(iter(_embedding_cache))
                del _embedding_cache[oldest_key]
            _embedding_cache[text_hash] = embedding.copy()

    return embedding


def create_embedding_vectors_batch(texts: list[str], use_cache: bool = True):
    """
    여러 텍스트를 배치로 임베딩 벡터로 변환 (리소스 효율적)
    
    배치 처리로 GPU 활용도 향상 및 처리 속도 개선
    
    Args:
        texts (list[str]): 임베딩할 텍스트 리스트
        use_cache (bool): 캐시 사용 여부 (기본값: True)
        
    Returns:
        list[numpy.ndarray]: 임베딩 벡터 리스트
    """
    if not texts:
        return []
    
    # 캐시 확인 및 미캐시된 텍스트만 필터링
    uncached_texts = []
    uncached_indices = []
    cached_embeddings = {}
    
    if use_cache:
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[idx] = None
                continue
            text_hash = _get_text_hash(text)
            with _embedding_cache_lock:
                if text_hash in _embedding_cache:
                    cached_embeddings[idx] = _embedding_cache[text_hash].copy()
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
    else:
        uncached_texts = [t for t in texts if t and t.strip()]
        uncached_indices = list(range(len(uncached_texts)))
    
    # 배치 임베딩 생성
    if uncached_texts:
        model = load_embedding_model()
        embeddings = model.encode(
            uncached_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,  # 배치 크기 최적화
        ).astype("float32")
        
        # 캐시 저장
        if use_cache:
            with _embedding_cache_lock:
                for text, embedding in zip(uncached_texts, embeddings):
                    text_hash = _get_text_hash(text)
                    if len(_embedding_cache) >= EMBEDDING_CACHE_SIZE:
                        oldest_key = next(iter(_embedding_cache))
                        del _embedding_cache[oldest_key]
                    _embedding_cache[text_hash] = embedding.copy()
        
        # 결과 매핑
        for idx, embedding in zip(uncached_indices, embeddings):
            cached_embeddings[idx] = embedding
    
    # 원래 순서대로 반환
    result = []
    for i in range(len(texts)):
        if i in cached_embeddings:
            result.append(cached_embeddings[i])
        else:
            result.append(None)
    
    return result

def create_embedding_from_image(image_bytes):
    """
    이미지를 직접 임베딩 벡터로 변환 (이미지 기반 검색용)
    
    TODO: AI 팀 구현 필요
    
    Args:
        image_bytes (bytes): 검색에 사용할 이미지 파일의 바이트 데이터
        
    Returns:
        numpy.ndarray: shape (EMBEDDING_DIMENSION,) 임베딩 벡터
        
    구현 가이드:
        1. CLIP 또는 유사한 멀티모달 모델 사용
        2. 이미지를 벡터로 변환
        3. 텍스트 임베딩과 같은 공간에 매핑되도록 처리
        4. 정규화 적용
    """
    caption = describe_image_with_llava(image_bytes)
    if not caption:
        raise ValueError("이미지 캡셔닝 결과가 비어 있습니다.")
    return create_embedding_vector(caption)


def warmup_models():
    """서버 기동 시 주요 모델을 미리 로드하여 콜드스타트를 줄임."""
    global models_warmed
    if models_warmed:
        return
    try:
        print("🔥 모델 워밍업 시작...")
        load_embedding_model()
        preprocess_text("모델 워밍업")
        models_warmed = True
        print("✅ 모델 워밍업 완료")
    except Exception as exc:
        print(f"⚠️ 모델 워밍업 실패: {exc}")

models_warmed = False
# 각 워커 시작 시 모델과 FAISS 미리 로드
# 예외가 발생해도 앱이 시작될 수 있도록 try-except 처리
try:
    initialize_faiss()
except Exception as e:
    print(f"⚠️ FAISS 초기화 실패 (앱은 계속 시작됩니다): {e}")
    import traceback
    traceback.print_exc()
    # 빈 인덱스로 시작 (예외 발생 시)
    if faiss_index is None:
        if FAISS_INDEX_TYPE.upper() == "HNSW":
            faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
            faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        else:
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        id_mapping = {}
        _faiss_initialized = True
        print(f"✅ 빈 FAISS 인덱스로 시작합니다 (CSV 임포트 API 호출 시 데이터 추가됨).")

try:
    warmup_models()
except Exception as e:
    print(f"⚠️ 모델 워밍업 실패 (앱은 계속 시작됩니다): {e}")
    import traceback
    traceback.print_exc()


@app.route('/health')
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'service': 'lostfound-ai-server',
        'faiss_vectors': faiss_index.ntotal if faiss_index else 0
    })

@app.route('/api/v1/embedding/create', methods=['POST'])
def create_embedding():
    """
    분실물 등록 시 임베딩 생성 및 FAISS 저장
    
    프로세스:
    1. 사용자가 등록한 이미지를 LLaVA로 분석하여 자연어 설명 생성
    2. (이미지 설명 + 사용자 입력 설명)을 BGE-M3로 임베딩 벡터로 변환
    3. 임베딩 벡터를 FAISS 인덱스에 저장
    4. MySQL item_id와 FAISS 인덱스 번호를 매핑하여 저장
    
    Spring에서 받는 것:
    - item_id: MySQL 분실물 ID (필수)
    - description: 사용자가 입력한 분실물 설명 (선택)
    - image: 분실물 이미지 파일 (선택)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_id: 원본 ID (확인용)
    - message: 결과 메시지
    """
    try:
        item_id = request.form.get('item_id')
        item_name = request.form.get('item_name', '')  # 분실물 제목
        raw_description = request.form.get('description', '')
        image_file = request.files.get('image')
        
        if not item_id:
            return jsonify({'success': False, 'message': 'item_id 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        # 1. 이미지 묘사 생성 (Qwen 기반) - 혁신적 개선된 프롬프트 사용
        image_description = ""
        caption_failed = False
        if image_file:
            image_bytes = image_file.read()
            image_description = describe_image_with_llava(image_bytes)
            if not image_description:
                caption_failed = True
                if raw_description:
                    image_description = raw_description.strip()
                    print(f"⚠️ 이미지 캡셔닝 실패, 원본 description 사용: {image_description[:150]}...")
                else:
                    print(f"⚠️ 이미지 캡셔닝 실패, description 없음")
            else:
                print(f"🖼️ 이미지 분석 완료: {image_description[:150]}...")
        
        # 2. 캡션 + 분실물 제목 + 사용자 설명 결합
        #    혁신적 개선: 사용자 입력도 검색 최적화 전처리 적용
        parts = []
        if image_description:
            parts.append(image_description)
        if item_name and item_name.strip():
            parts.append(item_name.strip())
        if raw_description and raw_description.strip():
            # 사용자 입력도 검색 최적화 전처리 적용
            processed_description = preprocess_text(
                raw_description.strip(),
                use_typo_correction=False,  # 사용자 입력은 맞춤법 교정 스킵
                optimize_for_search=True   # 검색 최적화 적용
            )
            if processed_description:
                parts.append(processed_description)
            else:
                parts.append(raw_description.strip())
        
        if not parts:
            return jsonify({'success': False, 'message': '설명 정보가 필요합니다.'}), 400
        
        # 원본 텍스트 결합
        raw_full_text = " ".join(parts).strip()
        
        # 3. 통합 전처리 (검색 시와 동일한 전처리 적용)
        #    저장 시와 검색 시 동일한 전처리를 적용하여 일관성 보장
        #    리소스 절약: 등록 시에는 맞춤법 교정을 선택적으로 사용
        final_text = preprocess_text(
            raw_full_text, 
            use_typo_correction=True,  # 등록 시에는 맞춤법 교정 사용
            optimize_for_search=True   # 검색 최적화 적용
        )
        if not final_text or len(final_text.strip()) == 0:
            # 전처리 실패 시 원본 사용 (공백 제거만)
            final_text = raw_full_text.strip()
        
        # 최종 임베딩 텍스트 로그 출력 (FAISS 저장 전)
        print(f"📝 [임베딩 텍스트] item_id={item_id}")
        print(f"   캡션: {image_description[:200] if image_description else '(없음)'}")
        print(f"   제목: {item_name if item_name else '(없음)'}")
        print(f"   설명: {raw_description[:200] if raw_description else '(없음)'}")
        print(f"   결합된 원본 텍스트: {raw_full_text}")
        print(f"   전처리 후 최종 임베딩 텍스트: {final_text}")
        print(f"   텍스트 길이: 원본={len(raw_full_text)}자, 최종={len(final_text)}자")
        
        # 4. 텍스트를 임베딩 벡터로 변환 (BGE-M3 사용)
        #    검색 시와 동일한 방식으로 임베딩 생성
        embedding_vector = create_embedding_vector(final_text)
        
        # 4. FAISS 인덱스에 벡터 추가 (스레드 안전하게 처리)
        should_save = False
        # 파일 잠금을 사용하여 멀티 워커 환경에서 안전하게 FAISS에 추가
        lock_file_path = os.path.join(FAISS_STORAGE_DIR, '.faiss_lock')
        os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
        
        with open(lock_file_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # 배타적 잠금
            try:
                with _faiss_lock:  # 같은 워커 내 스레드 동기화도 유지
                    faiss_index.add(np.array([embedding_vector]))
                    faiss_idx = faiss_index.ntotal - 1
                    # 5. FAISS 인덱스 번호 ↔ MySQL item_id 매핑 저장
                    id_mapping[faiss_idx] = int(item_id)
                    # 저장 빈도 제어: N개마다 저장하여 디스크 I/O 최적화
                    global _pending_save_count
                    _pending_save_count += 1
                    if _pending_save_count >= _save_batch_size:
                        should_save = True
                        _pending_save_count = 0
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        
        # 6. FAISS 인덱스 및 매핑 정보를 디스크에 저장 (영속성)
        # 배치 단위로 저장하여 디스크 I/O 최적화
        if should_save:
            save_faiss()
        
        print(f"✅ 임베딩 생성 완료: item_id={item_id}, faiss_idx={faiss_idx}, 벡터 차원={len(embedding_vector)}")
        
        return jsonify({
            'success': True,
            'item_id': int(item_id),
            'message': f'임베딩 생성 완료 (FAISS 인덱스: {faiss_idx})'
        })
        
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/create-batch', methods=['POST'])
def create_embeddings_batch():
    """
    분실물 등록 시 배치 임베딩 생성 및 FAISS 저장 (성능 최적화)
    
    여러 아이템을 한 번에 처리하여 네트워크 오버헤드 감소 및 처리 속도 향상
    
    Spring에서 받는 것:
    - items: 아이템 리스트 (각 아이템은 다음 필드 포함)
      - item_id: MySQL 분실물 ID (필수)
      - description: 사용자가 입력한 분실물 설명 (선택)
      - image_url: 이미지 URL (선택)
      - image: 이미지 파일 (image_url이 없을 경우, 선택)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - results: 각 아이템별 결과 리스트
      - item_id: 원본 ID
      - success: 성공 여부
      - message: 결과 메시지
      - faiss_idx: FAISS 인덱스 번호 (성공 시)
    """
    try:
        items_data = request.form.get('items')
        if not items_data:
            return jsonify({'success': False, 'message': 'items 데이터 필요'}), 400
        
        try:
            items = json.loads(items_data)
        except json.JSONDecodeError:
            return jsonify({'success': False, 'message': 'items JSON 파싱 실패'}), 400
        
        if not isinstance(items, list) or len(items) == 0:
            return jsonify({'success': False, 'message': 'items는 비어있지 않은 리스트여야 합니다'}), 400
        
        # FAISS 인덱스 초기화 확인
        if not _faiss_initialized:
            initialize_faiss()
        
        results = []
        successful_count = 0
        
        # 배치 처리: 여러 이미지를 병렬로 처리
        def process_item(item):
            """단일 아이템 처리"""
            item_id = item.get('item_id')
            item_name = item.get('item_name', '')  # 분실물 제목
            raw_description = item.get('description', '')
            image_url = item.get('image_url', '')
            
            if not item_id:
                return {
                    'item_id': None,
                    'success': False,
                    'message': 'item_id 필요'
                }
            
            try:
                # 1. 이미지 다운로드 및 캡셔닝 (재시도 로직 포함)
                image_description = ""
                caption_failed = False
                if image_url:
                    # Qwen 7B 모델 리소스 고려: 재시도 1회로 감소, 타임아웃 단축
                    max_retries = 1  # 2 -> 1로 감소하여 리소스 절약
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(
                                image_url, 
                                timeout=(3, 10),  # 타임아웃 단축: (5,15) -> (3,10)
                                stream=True,
                                headers={'User-Agent': 'Mozilla/5.0'}  # 일부 서버에서 필요
                            )
                            response.raise_for_status()
                            image_bytes = response.content
                            if len(image_bytes) > 20 * 1024 * 1024:  # 20MB 제한
                                raise ValueError("이미지 파일이 너무 큼")
                            if len(image_bytes) == 0:
                                raise ValueError("이미지 파일이 비어있음")
                            image_description = describe_image_with_llava(image_bytes)
                            if image_description:
                                break  # 성공 시 루프 종료
                            else:
                                caption_failed = True
                                break  # 캡셔닝 실패 시 루프 종료
                        except Exception as e:
                            caption_failed = True
                            print(f"⚠️ 이미지 다운로드/캡셔닝 실패 (item_id={item_id}): {e}")
                            # 재시도 없이 바로 텍스트로 진행하여 리소스 절약
                
                # 캡셔닝 실패 시 원본 description 사용
                if caption_failed and not image_description and raw_description:
                    image_description = raw_description.strip()
                
                # 2. 캡션 + 분실물 제목 + 사용자 설명 결합 및 전처리
                parts = []
                if image_description:
                    parts.append(image_description)
                if item_name and item_name.strip():
                    parts.append(item_name.strip())
                if raw_description and raw_description.strip():
                    parts.append(raw_description.strip())
                
                if not parts:
                    return {
                        'item_id': int(item_id),
                        'success': False,
                        'message': '설명 정보가 필요합니다'
                    }
                
                raw_full_text = " ".join(parts).strip()
                final_text = preprocess_text(
                    raw_full_text,
                    use_typo_correction=True,
                    optimize_for_search=True
                )
                if not final_text or len(final_text.strip()) == 0:
                    final_text = raw_full_text.strip()
                
                # 최종 임베딩 텍스트 로그 출력 (FAISS 저장 전)
                print(f"📝 [배치 임베딩 텍스트] item_id={item_id}")
                print(f"   캡션: {image_description[:200] if image_description else '(없음)'}")
                print(f"   제목: {item_name if item_name else '(없음)'}")
                print(f"   설명: {raw_description[:200] if raw_description else '(없음)'}")
                print(f"   결합된 원본 텍스트: {raw_full_text}")
                print(f"   전처리 후 최종 임베딩 텍스트: {final_text}")
                print(f"   텍스트 길이: 원본={len(raw_full_text)}자, 최종={len(final_text)}자")
                
                # 3. 임베딩 벡터 생성 (캐시 활용)
                embedding_vector = create_embedding_vector(final_text, use_cache=True)
                
                # 4. FAISS 인덱스에 벡터 추가 (멀티 워커 환경에서 안전하게 처리)
                faiss_idx = None
                before_count = faiss_index.ntotal
                # 파일 잠금을 사용하여 멀티 워커 환경에서 안전하게 FAISS에 추가
                lock_file_path = os.path.join(FAISS_STORAGE_DIR, '.faiss_lock')
                os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
                
                with open(lock_file_path, 'w') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # 배타적 잠금
                    try:
                        with _faiss_lock:  # 같은 워커 내 스레드 동기화도 유지
                            faiss_index.add(np.array([embedding_vector]))
                            faiss_idx = faiss_index.ntotal - 1
                            id_mapping[faiss_idx] = int(item_id)
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
                return {
                    'item_id': int(item_id),
                    'success': True,
                    'message': f'임베딩 생성 완료',
                    'faiss_idx': faiss_idx
                }
                
            except Exception as e:
                print(f"❌ 배치 임베딩 생성 실패 (item_id={item_id}): {str(e)}")
                return {
                    'item_id': int(item_id) if item_id else None,
                    'success': False,
                    'message': str(e)
                }
        
        # 순차 처리로 변경 (메모리 안정성 확보)
        # Qwen 7B 모델은 메모리를 많이 사용하므로 동시 처리 시 OOM 발생 가능
        # 순차 처리로 변경하여 안정성 확보 (성능은 다소 저하되지만 메모리 안정성 우선)
        for item in items:
            result = process_item(item)
            results.append(result)
            if result.get('success'):
                successful_count += 1
            
            # 각 아이템 처리 후 메모리 정리 (선택적, 성능 저하 가능하므로 주석 처리)
            # 필요시 주석 해제하여 메모리 정리 활성화
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        
        # FAISS 인덱스 및 매핑 정보를 디스크에 저장 (배치 완료 후 한 번만)
        # 배치 API는 항상 저장하여 데이터 손실 방지
        with _faiss_lock:
            global _pending_save_count
            _pending_save_count = 0  # 배치 저장 시 카운터 리셋
        save_faiss()
        
        print(f"✅ 배치 임베딩 생성 완료: {successful_count}/{len(items)}개 성공")
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(items),
                'successful': successful_count,
                'failed': len(items) - successful_count
            }
        })
        
    except Exception as e:
        print(f"❌ 배치 임베딩 생성 실패: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/search', methods=['POST'])
def search_embedding():
    """
    자연어 검색: 텍스트 쿼리로 유사한 분실물 검색
    
    프로세스:
    1. 사용자의 검색어를 BGE-M3로 임베딩 벡터로 변환
    2. FAISS에서 코사인 유사도 기반 Top-K 검색
    3. 유사도가 높은 순서대로 MySQL item_id 리스트 반환
    
    Spring에서 받는 것:
    - query: 자연어 검색어 (필수)
      예) "지하철에서 잃어버린 검은 지갑", "강남역에서 발견한 아이폰"
    - top_k: 반환할 개수 (선택, 기본 10)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_ids: 유사도 높은 순서대로 정렬된 MySQL item_id 리스트
    
    TODO: AI 팀 추가 구현 사항
    - 날짜/장소 필터링 가중치 적용
    - 하이브리드 검색 (키워드 + 시맨틱)
    """
    try:
        print(f"🔍 검색 요청 수신: Content-Type={request.content_type}, Method={request.method}")
        
        data = request.get_json()
        if data is None:
            print(f"❌ 요청 본문이 None입니다. Content-Type: {request.content_type}")
            return jsonify({'success': False, 'message': '요청 본문이 비어있습니다'}), 400
        
        print(f"📥 요청 데이터 수신: {data}")
        
        raw_query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        print(f"🔍 검색 파라미터: raw_query='{raw_query}', top_k={top_k}")
        
        if not raw_query or not raw_query.strip():
            print(f"❌ 검색어가 비어있습니다: raw_query='{raw_query}'")
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # 검색 쿼리 전처리 (리소스 절약: 검색 시에는 맞춤법 교정 선택적)
        query = preprocess_text(
            raw_query,
            use_typo_correction=False,  # 검색 시에는 맞춤법 교정 스킵하여 리소스 절약
            optimize_for_search=True    # 검색 최적화 적용
        )
        if not query:
            query = raw_query.strip()
        
        print(f"📝 전처리 후 검색어: '{query}' (원본: '{raw_query}')")
        
        if not query:
            print(f"❌ 전처리 후에도 검색어가 비어있습니다")
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        # FAISS 인덱스가 비어있으면 빈 결과 반환
        if faiss_index is None or faiss_index.ntotal == 0:
            print(f"❌ FAISS 인덱스 비어있음: ntotal={faiss_index.ntotal if faiss_index else 0}, id_mapping={len(id_mapping)}")
            return jsonify({'success': True, 'item_ids': [], 'scores': []})
        
        print(f"✅ FAISS 인덱스 상태: ntotal={faiss_index.ntotal}, id_mapping={len(id_mapping)}")
        
        # 1. 검색어를 임베딩 벡터로 변환 (BGE-M3 사용, 캐시 활용)
        print(f"🔄 검색어 임베딩 벡터 변환 시작: query='{query}'")
        query_vector = create_embedding_vector(query, use_cache=True)
        print(f"✅ 임베딩 벡터 생성 완료: shape={query_vector.shape}")
        
        # 2. FAISS에서 코사인 유사도 기반 검색
        # top_k는 최대 반환 개수로만 사용 (상한선)
        # 유사도 임계값 이상인 결과를 최대 top_k개까지 반환
        k = min(max(top_k * 3, top_k + 50), faiss_index.ntotal)  # 충분히 많이 가져와서 필터링
        if k == 0:
            print(f"❌ k=0: top_k={top_k}, ntotal={faiss_index.ntotal}")
            return jsonify({'success': True, 'item_ids': [], 'scores': []})
        
        print(f"📊 검색 파라미터: top_k={top_k} (최대 반환 개수), k={k} (검색 범위), ntotal={faiss_index.ntotal}, 임계값={SIMILARITY_THRESHOLD}")
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = max(HNSW_EF_SEARCH, k * 2)
        
        # 검색 실행
        print(f"🔍 FAISS 검색 실행: k={k}, ntotal={faiss_index.ntotal}")
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        print(f"✅ FAISS 검색 완료: distances shape={distances.shape}, indices shape={indices.shape}")
        
        valid_results = len([idx for idx in indices[0] if int(idx) != -1])
        if valid_results == 0:
            print(f"❌ FAISS 검색 결과 없음: k={k}, ntotal={faiss_index.ntotal}, id_mapping={len(id_mapping)}")
            return jsonify({'success': True, 'item_ids': [], 'scores': []})
        
        print(f"📊 유효한 검색 결과: {valid_results}개")
        
        # FAISS 인덱스 번호 → MySQL item_id 변환 및 유사도 임계값 필터링
        # 유사도 임계값 이상인 결과만 동적으로 수집 (top_k는 최대 개수로만 사용)
        item_ids = []
        scores = []
        threshold_passed = 0
        threshold_failed = 0
        mapping_missing = 0
        
        for idx, dist in zip(indices[0], distances[0]):
            # top_k를 초과하면 즉시 중단 (최대 개수 제한)
            if len(item_ids) >= top_k:
                break
                
            if int(idx) != -1:
                if int(idx) in id_mapping:
                    score = float(dist)  # IndexFlatIP이므로 내적 값 (높을수록 유사)
                    # 유사도 임계값 이상인 결과만 포함 (동적 필터링)
                    # BGE-M3는 정규화된 임베딩을 사용하므로 내적 값은 대략 0.3~0.95 범위
                    if score >= SIMILARITY_THRESHOLD:
                        threshold_passed += 1
                        item_ids.append(id_mapping[int(idx)])
                        scores.append(score)
                    else:
                        threshold_failed += 1
                        # 임계값 미만인 경우 로그 (디버깅용, 처음 몇 개만)
                        if threshold_failed <= 5:
                            print(f"   임계값 미만: item_id={id_mapping[int(idx)]}, score={score:.4f} < {SIMILARITY_THRESHOLD}")
                else:
                    mapping_missing += 1
        
        # 안전장치: 모든 scores를 Python float로 강제 변환 (numpy 타입 방지)
        safe_scores = []
        for s in scores:
            try:
                safe_scores.append(float(s))  # numpy float32, float64 등 모든 숫자 타입을 Python float로 변환
            except (TypeError, ValueError):
                # 변환 실패 시 0.0으로 대체 (안전장치)
                safe_scores.append(0.0)
        
        result = {
            'success': True,
            'item_ids': item_ids,  # 유사도 임계값 이상인 결과만 반환 (최대 top_k개)
            'scores': safe_scores
        }
        
        print(f"✅ 검색 완료 및 응답 반환: item_ids={len(result['item_ids'])}, scores={len(result['scores'])}")
        print(f"   임계값 통과: {threshold_passed}개, 임계값 미만: {threshold_failed}개, 매핑 없음: {mapping_missing}개")
        if result['item_ids']:
            print(f"   상위 5개 item_ids: {result['item_ids'][:5]}")
        if result['scores']:
            print(f"   상위 5개 scores: {[f'{s:.4f}' for s in result['scores'][:5]]}")
        print(f"   최대 반환 개수(top_k): {top_k}, 실제 반환: {len(result['item_ids'])}개 (유사도 임계값 이상만)")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 검색 실패: {str(e)}")
        import traceback
        print(f"   상세 에러:\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/search-by-image', methods=['POST'])
def search_by_image():
    """
    이미지 기반 검색: 사용자가 업로드한 이미지와 유사한 분실물 검색
    
    프로세스:
    1. 업로드된 이미지를 LLaVA 또는 CLIP으로 임베딩 벡터로 변환
    2. FAISS에서 유사도 검색
    3. 유사도 높은 순서대로 MySQL item_id 리스트 반환
    
    Spring에서 받는 것:
    - image: 검색에 사용할 이미지 파일 (필수)
    - top_k: 반환할 개수 (선택, 기본 10)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_ids: 유사도 높은 순서대로 정렬된 MySQL item_id 리스트
    
    TODO: AI 팀 구현 필요
    - create_embedding_from_image() 함수 구현
    - 멀티모달 모델 (CLIP 등) 사용
    """
    try:
        image_file = request.files.get('image')
        top_k = int(request.form.get('top_k', 10))
        
        if not image_file:
            return jsonify({'success': False, 'message': '이미지 파일 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': [], 'scores': []})
        
        # 1. 이미지를 임베딩 벡터로 변환
        #    AI 팀: create_embedding_from_image() 함수 구현 필요
        image_bytes = image_file.read()
        try:
            query_vector = create_embedding_from_image(image_bytes)
        except ValueError as err:
            return jsonify({'success': False, 'message': str(err)}), 400
        
        # 2. FAISS에서 유사도 검색
        # top_k는 최대 반환 개수로만 사용 (상한선)
        # 유사도 임계값 이상인 결과를 최대 top_k개까지 반환
        k = min(max(top_k * 3, top_k + 50), faiss_index.ntotal)  # 충분히 많이 가져와서 필터링
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = max(HNSW_EF_SEARCH, k * 2)
        
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        
        # 3. FAISS 인덱스 번호 → MySQL item_id 변환 및 유사도 임계값 필터링
        # 유사도 임계값 이상인 결과만 동적으로 수집 (top_k는 최대 개수로만 사용)
        item_ids = []
        scores = []
        threshold_passed = 0
        threshold_failed = 0
        
        for idx, dist in zip(indices[0], distances[0]):
            # top_k를 초과하면 즉시 중단 (최대 개수 제한)
            if len(item_ids) >= top_k:
                break
                
            if int(idx) != -1 and int(idx) in id_mapping:
                score = float(dist)  # numpy float32를 Python float로 명시적 변환
                # 유사도 임계값 이상인 결과만 포함 (동적 필터링)
                if score >= SIMILARITY_THRESHOLD:
                    threshold_passed += 1
                    item_ids.append(id_mapping[int(idx)])
                    scores.append(score)
                else:
                    threshold_failed += 1
        
        # 안전장치: 모든 scores를 Python float로 강제 변환 (numpy 타입 방지)
        safe_scores = []
        for s in scores:
            try:
                safe_scores.append(float(s))  # numpy float32, float64 등 모든 숫자 타입을 Python float로 변환
            except (TypeError, ValueError):
                # 변환 실패 시 0.0으로 대체 (안전장치)
                safe_scores.append(0.0)
        
        print(f"🔍 이미지 검색 완료: 최대 반환 개수(top_k)={top_k}, 실제 반환={len(item_ids)}개")
        print(f"   임계값 통과: {threshold_passed}개, 임계값 미만: {threshold_failed}개, 임계값={SIMILARITY_THRESHOLD}")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids,  # 유사도 임계값 이상인 결과만 반환 (최대 top_k개)
            'scores': safe_scores
        })
        
    except Exception as e:
        print(f"❌ 이미지 검색 실패: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/search-with-filters', methods=['POST'])
def search_with_filters():
    """
    필터링 가중치를 적용한 고급 검색
    
    프로세스:
    1. 자연어 검색으로 유사한 분실물 후보 추출
    2. 날짜, 장소 필터에 대한 가중치 적용
    3. 재정렬하여 결과 반환
    
    Spring에서 받는 것:
    - query: 자연어 검색어 (필수)
    - top_k: 반환할 개수 (선택, 기본 10)
    - filters: 필터링 조건 (선택)
      - location: 장소 (예: "강남역")
      - start_date: 시작 날짜 (예: "2025-01-01")
      - end_date: 종료 날짜 (예: "2025-01-31")
    - weights: 가중치 설정 (선택)
      - semantic: 시맨틱 유사도 가중치 (0~1, 기본 0.7)
      - location: 장소 일치 가중치 (0~1, 기본 0.2)
      - date: 날짜 일치 가중치 (0~1, 기본 0.1)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_ids: 재정렬된 MySQL item_id 리스트
    
    TODO: AI 팀 구현 필요
    - 필터링 로직 구현
    - 가중치 기반 스코어링 시스템
    - Spring에서 메타데이터 함께 전달받는 방식 고려
    
    참고:
    현재는 기본 검색만 수행하며, 필터링은 Spring 단에서 처리됨
    향후 AI 단에서 필터링 가중치를 적용한 고급 검색 구현 가능
    """
    try:
        data = request.get_json()
        raw_query = data.get('query', '')
        if not raw_query or not raw_query.strip():
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # 검색 쿼리 전처리 (리소스 절약)
        query = preprocess_text(
            raw_query,
            use_typo_correction=False,  # 검색 시에는 맞춤법 교정 스킵
            optimize_for_search=True
        )
        if not query:
            query = raw_query.strip()
        
        top_k = data.get('top_k', 10)
        filters = data.get('filters', {})
        weights = data.get('weights', {
            'semantic': 0.7,
            'location': 0.2,
            'date': 0.1
        })
        
        if not query:
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 기본 시맨틱 검색 (캐시 활용)
        query_vector = create_embedding_vector(query, use_cache=True)
        k = min(top_k * 3, faiss_index.ntotal)  # 더 많이 가져와서 필터링
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = max(HNSW_EF_SEARCH, k * 2)
        
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        
        # 2. 초기 후보 추출
        item_ids = []
        for idx in indices[0]:
            if int(idx) in id_mapping:
                item_ids.append(id_mapping[int(idx)])
        
        # TODO: AI 팀 구현 필요
        # 3. 필터링 가중치 적용 및 재정렬
        # - Spring에서 각 item의 메타데이터(장소, 날짜 등)를 받아야 함
        # - 또는 Flask에서 별도 DB 연결하여 메타데이터 조회
        # - 가중치 기반 스코어 계산: score = w1*sim + w2*loc_match + w3*date_match
        # - 스코어 기반 재정렬
        
        print(f"🔍 필터링 검색 완료: query='{query[:30]}...', 결과={len(item_ids[:top_k])}개")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids[:top_k]
        })
        
    except Exception as e:
        print(f"❌ 필터링 검색 실패: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/admin/sync-with-db', methods=['POST'])
def sync_faiss_with_db():
    """
    Admin API: DB와 FAISS 동기화
    
    DB에는 없지만 FAISS에는 있는 항목들을 찾아서 FAISS에서 삭제합니다.
    (고아 데이터 정리)
    
    Spring에서 받는 것:
    - db_item_ids: DB에 실제로 존재하는 모든 item_id 리스트 (필수)
      예: [1, 2, 3, 5, 7, ...]
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - total_faiss_items: FAISS에 있는 전체 항목 수
    - total_db_items: DB에 있는 전체 항목 수
    - orphaned_items: FAISS에는 있지만 DB에는 없는 항목 리스트
    - deleted_count: 실제로 삭제된 항목 수
    """
    try:
        data = request.get_json()
        db_item_ids = data.get('db_item_ids', [])
        
        if not isinstance(db_item_ids, list):
            return jsonify({'success': False, 'message': 'db_item_ids는 리스트여야 합니다'}), 400
        
        # FAISS 인덱스 초기화 확인
        if not _faiss_initialized:
            initialize_faiss()
        
        if faiss_index is None:
            return jsonify({
                'success': False,
                'message': 'FAISS 인덱스가 초기화되지 않았습니다'
            }), 500
        
        # DB item_id를 set으로 변환 (빠른 조회를 위해)
        db_item_set = set(int(item_id) for item_id in db_item_ids)
        
        # FAISS에 있는 모든 item_id 추출
        faiss_item_ids = set(id_mapping.values())
        
        # 고아 데이터 찾기: FAISS에는 있지만 DB에는 없는 항목들
        orphaned_item_ids = faiss_item_ids - db_item_set
        
        print(f"🔍 동기화 시작:")
        print(f"   DB 항목 수: {len(db_item_set)}")
        print(f"   FAISS 항목 수: {len(faiss_item_ids)}")
        print(f"   고아 데이터: {len(orphaned_item_ids)}개")
        
        if len(orphaned_item_ids) == 0:
            print("✅ 동기화 완료: 고아 데이터 없음")
            return jsonify({
                'success': True,
                'total_faiss_items': len(faiss_item_ids),
                'total_db_items': len(db_item_set),
                'orphaned_items': [],
                'deleted_count': 0,
                'message': '고아 데이터가 없습니다. 동기화 완료.'
            })
        
        # 고아 데이터를 FAISS에서 삭제
        deleted_count = 0
        deleted_item_ids = []
        
        with _faiss_lock:
            for orphaned_item_id in orphaned_item_ids:
                try:
                    # 해당 item_id의 faiss_idx 찾기
                    faiss_indices_to_delete = [k for k, v in id_mapping.items() if v == orphaned_item_id]
                    
                    if len(faiss_indices_to_delete) == 0:
                        continue
                    
                    # FAISS 인덱스에서 벡터 삭제
                    if hasattr(faiss_index, 'remove_ids'):
                        try:
                            ids_to_remove = np.array(faiss_indices_to_delete, dtype=np.int64)
                            faiss_index.remove_ids(ids_to_remove)
                            deleted_count += len(faiss_indices_to_delete)
                        except Exception as e:
                            print(f"⚠️  FAISS 벡터 삭제 실패 (item_id={orphaned_item_id}): {str(e)}")
                    
                    # id_mapping에서 제거
                    for faiss_idx in faiss_indices_to_delete:
                        if faiss_idx in id_mapping:
                            del id_mapping[faiss_idx]
                    
                    deleted_item_ids.append(orphaned_item_id)
                    
                except Exception as e:
                    print(f"⚠️  항목 삭제 실패 (item_id={orphaned_item_id}): {str(e)}")
                    continue
        
        # 영속성 저장
        save_faiss()
        
        print(f"✅ 동기화 완료: {deleted_count}개 벡터 삭제됨 (고아 데이터 {len(orphaned_item_ids)}개)")
        
        return jsonify({
            'success': True,
            'total_faiss_items': len(faiss_item_ids),
            'total_db_items': len(db_item_set),
            'orphaned_items': sorted(list(orphaned_item_ids)),
            'deleted_count': deleted_count,
            'deleted_item_ids': sorted(deleted_item_ids),
            'message': f'{deleted_count}개 벡터가 삭제되었습니다.'
        })
        
    except Exception as e:
        print(f"❌ 동기화 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/delete/<int:item_id>', methods=['DELETE'])
def delete_embedding(item_id):
    """
    분실물 삭제 시 임베딩 제거
    
    프로세스:
    1. id_mapping에서 해당 item_id의 faiss_idx 찾기
    2. FAISS 인덱스에서 벡터 물리적 삭제 (HNSW의 경우 remove_ids 사용)
    3. id_mapping에서 제거
    4. 영속성 저장
    
    Spring에서 받는 것:
    - item_id: 삭제할 분실물의 MySQL ID
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - deleted_count: 삭제된 벡터 개수
    """
    try:
        with _faiss_lock:
            # 1. id_mapping에서 해당 item_id의 faiss_idx 찾기
            faiss_indices_to_delete = [k for k, v in id_mapping.items() if v == item_id]
            
            if len(faiss_indices_to_delete) == 0:
                print(f"⚠️  삭제 시도: item_id={item_id}, 하지만 매핑에서 찾을 수 없음 (이미 삭제되었거나 존재하지 않음)")
                return jsonify({'success': True, 'deleted_count': 0})
            
            # 2. FAISS 인덱스에서 벡터 물리적 삭제
            deleted_count = 0
            if hasattr(faiss_index, 'remove_ids'):
                # HNSW 인덱스: remove_ids() 메서드 사용
                try:
                    # FAISS의 remove_ids는 numpy array를 받음
                    ids_to_remove = np.array(faiss_indices_to_delete, dtype=np.int64)
                    faiss_index.remove_ids(ids_to_remove)
                    deleted_count = len(faiss_indices_to_delete)
                    print(f"🗑️  FAISS에서 벡터 삭제 완료: item_id={item_id}, faiss_indices={faiss_indices_to_delete}")
                except Exception as e:
                    print(f"⚠️  FAISS 벡터 삭제 실패 (id_mapping만 제거): {str(e)}")
                    # FAISS 삭제 실패해도 id_mapping은 제거
            else:
                # Flat 인덱스: 직접 삭제 불가능, id_mapping에서만 제거
                print(f"⚠️  Flat 인덱스는 직접 삭제 불가능, id_mapping에서만 제거: item_id={item_id}")
            
            # 3. id_mapping에서 제거
            for faiss_idx in faiss_indices_to_delete:
                if faiss_idx in id_mapping:
                    del id_mapping[faiss_idx]
                    deleted_count = max(deleted_count, 1)  # 최소 1개는 삭제됨
        
        # 4. 영속성 저장
        save_faiss()
        
        print(f"🗑️  삭제 완료: item_id={item_id}, 제거된 벡터={deleted_count}개 (FAISS 인덱스: {faiss_index.ntotal}개 남음)")
        
        return jsonify({'success': True, 'deleted_count': deleted_count})
        
    except Exception as e:
        print(f"❌ 삭제 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

# Gunicorn으로 실행할 때도 FAISS 초기화는 warmup_models()에서 수행됨
# 각 워커 프로세스가 시작될 때마다 초기화됨

if __name__ == '__main__':
    # 개발 모드에서 직접 실행할 때만 사용
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
