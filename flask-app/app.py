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

from services.captioning import generate_caption
from services.text_processing import preprocess_text

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

# ========== 전역 변수 ==========
faiss_index = None
id_mapping = {}  # FAISS 인덱스 번호 -> MySQL item_id 매핑
embedding_model: Optional[SentenceTransformer] = None
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
_faiss_initialized = False
_model_loaded = False
_faiss_lock = threading.Lock()  # FAISS 인덱스 접근 동기화

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
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        # 기존 인덱스 로드 (타입 자동 감지)
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, 'rb') as f:
            id_mapping = pickle.load(f)
        index_type_name = type(faiss_index).__name__
        print(f"✅ FAISS 인덱스 로드: {faiss_index.ntotal}개 벡터 (타입: {index_type_name})")
        
        # HNSW 인덱스인 경우 ef_search 설정
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
            print(f"   HNSW 파라미터 설정: ef_search={HNSW_EF_SEARCH}")
        
        # 인덱스 타입 불일치 경고 (설정과 다른 경우)
        if FAISS_INDEX_TYPE.upper() == "HNSW" and "Flat" in index_type_name and "HNSW" not in index_type_name:
            print(f"⚠️ 경고: 설정은 HNSW이지만 기존 인덱스는 {index_type_name}입니다.")
            print(f"   기존 인덱스를 사용합니다. 새 인덱스를 원하면 기존 파일을 삭제하세요.")
        elif FAISS_INDEX_TYPE.upper() == "FLAT" and "HNSW" in index_type_name:
            print(f"⚠️ 경고: 설정은 Flat이지만 기존 인덱스는 {index_type_name}입니다.")
            print(f"   기존 인덱스를 사용합니다. 새 인덱스를 원하면 기존 파일을 삭제하세요.")
    else:
        # 인덱스 타입에 따라 선택
        if FAISS_INDEX_TYPE.upper() == "HNSW":
            # HNSW 인덱스: 대량 데이터 검색 최적화 (근사 최근접 이웃)
            # IndexHNSWFlat: 내적 기반 + HNSW 그래프 구조
            faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, HNSW_M)
            faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
            print(f"✅ HNSW FAISS 인덱스 생성 (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, ef_search={HNSW_EF_SEARCH})")
        else:
            # Flat 인덱스: 정확한 검색 (소량 데이터용)
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            print("✅ Flat FAISS 인덱스 생성 (정확한 검색)")
        
        id_mapping = {}
    
    _faiss_initialized = True

def save_faiss():
    """FAISS 스냅샷 저장"""
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAPPING_PATH, 'wb') as f:
        pickle.dump(id_mapping, f)
    print(f"💾 FAISS 저장: {faiss_index.ntotal}개 벡터")

# ========== AI 팀이 구현할 함수들 (현재는 더미) ==========


def load_embedding_model() -> SentenceTransformer:
    """SentenceTransformer 모델을 1회 로드"""
    global embedding_model, _model_loaded
    if embedding_model is None or not _model_loaded:
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

def create_embedding_vector(text, is_query: bool = False):
    """
    BGE-M3 모델을 사용하여 텍스트를 임베딩 벡터로 변환
    
    Args:
        text (str): 임베딩할 텍스트
        is_query (bool): 검색 쿼리인지 여부 (기본값: False)
                        True인 경우 instruction 프리픽스가 이미 포함되어 있음
    
    Returns:
        numpy.ndarray: shape (EMBEDDING_DIMENSION,) 임베딩 벡터
    """
    if not text or not text.strip():
        raise ValueError("임베딩할 텍스트가 비어 있습니다.")

    model = load_embedding_model()
    
    # BGE-M3는 instruction을 활용하면 검색 성능이 향상됨
    # 저장 시: "이 문장을 기억합니다: " 프리픽스 사용
    # 검색 시: "이 문장을 검색합니다: " 프리픽스 사용 (이미 적용됨)
    if not is_query:
        # 저장 시 instruction 프리픽스 추가
        text_with_instruction = f"이 문장을 기억합니다: {text}"
    else:
        # 검색 시는 이미 프리픽스가 포함되어 있음
        text_with_instruction = text
    
    embedding = model.encode(
        [text_with_instruction],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")

    if embedding.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"임베딩 차원 불일치: 기대값={EMBEDDING_DIMENSION}, 실제값={embedding.shape[0]}"
        )

    return embedding

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
    return create_embedding_vector(caption, is_query=False)


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
initialize_faiss()
warmup_models()


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
    - item_name: 분실물 제목 (필수) - 예: "분홍색 지갑", "검은색 가방"
    - description: 사용자가 입력한 분실물 설명 (선택)
    - image: 분실물 이미지 파일 (선택)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_id: 원본 ID (확인용)
    - message: 결과 메시지
    """
    try:
        item_id = request.form.get('item_id')
        item_name = request.form.get('item_name', '')  # 분실물 제목 추가
        raw_description = request.form.get('description', '')
        image_file = request.files.get('image')
        
        if not item_id:
            return jsonify({'success': False, 'message': 'item_id 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        # 1. 이미지 묘사 생성 (Qwen 기반)
        image_description = ""
        if image_file:
            image_bytes = image_file.read()
            image_description = describe_image_with_llava(image_bytes)
            if not image_description and raw_description:
                image_description = raw_description.strip()
            print(f"🖼️  이미지 분석 완료 (원본): {image_description[:100]}...")
        
        # 2. 분실물 제목 + 이미지 묘사 + 사용자 설명 결합 (원본 텍스트로 결합)
        #    예) "분홍색 지갑. 빨간색 가죽 지갑입니다. 신촌역 3번 출구에서 발견했습니다."
        parts = []
        # 분실물 제목을 가장 먼저 추가 (검색 시 가장 중요)
        if item_name and item_name.strip():
            parts.append(item_name.strip())
        if image_description:
            parts.append(image_description)
        if raw_description and raw_description.strip():
            parts.append(raw_description.strip())
        
        if not parts:
            return jsonify({'success': False, 'message': '설명 정보가 필요합니다.'}), 400
        
        # 원본 텍스트 결합
        raw_full_text = " ".join(parts).strip()
        
        # 3. 통합 전처리 (저장 시에는 맞춤법 교정 사용)
        final_text = preprocess_text(raw_full_text, use_typo_correction=True)
        if not final_text or len(final_text.strip()) == 0:
            # 전처리 실패 시 원본 사용
            final_text = raw_full_text.strip()
        
        # 4. 임베딩 벡터 생성 (저장 시이므로 is_query=False)
        embedding_vector = create_embedding_vector(final_text, is_query=False)
        
        # 4. FAISS 인덱스에 벡터 추가 (스레드 안전하게 처리)
        with _faiss_lock:
            faiss_index.add(np.array([embedding_vector]))
            faiss_idx = faiss_index.ntotal - 1
            # 5. FAISS 인덱스 번호 ↔ MySQL item_id 매핑 저장
            id_mapping[faiss_idx] = int(item_id)
        
        # 6. FAISS 인덱스 및 매핑 정보를 디스크에 저장 (영속성)
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
      - item_name: 분실물 제목 (필수) - 예: "분홍색 지갑"
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
            item_name = item.get('item_name', '')  # 분실물 제목 추가
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
                if image_url:
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(
                                image_url, 
                                timeout=(5, 15),  # (연결 타임아웃, 읽기 타임아웃)
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
                            # 디버깅: 이미지 캡셔닝 성공 로그 (처음 10개만)
                            if int(item_id) <= 10:
                                print(f"✅ 이미지 캡셔닝 성공 (item_id={item_id}): '{image_description[:100]}'")
                            break  # 성공 시 루프 종료
                        except Exception as e:
                            if attempt == max_retries - 1:
                                print(f"⚠️ 이미지 다운로드/캡셔닝 실패 (item_id={item_id}, 시도 {attempt+1}/{max_retries}): {e}")
                                # 디버깅: 이미지 캡셔닝 실패 로그 (처음 10개만)
                                if int(item_id) <= 10:
                                    print(f"   이미지 URL: {image_url}")
                            else:
                                time.sleep(0.5 * (attempt + 1))  # 지수 백오프
                            # 마지막 시도 실패 시 텍스트로 진행
                else:
                    # 디버깅: 이미지 URL이 없는 경우 (처음 10개만)
                    if int(item_id) <= 10:
                        print(f"⚠️ 이미지 URL 없음 (item_id={item_id})")
                
                # 2. 분실물 제목 + 이미지 묘사 + 사용자 설명 결합
                parts = []
                # 분실물 제목을 가장 먼저 추가 (검색 시 가장 중요)
                if item_name and item_name.strip():
                    parts.append(item_name.strip())
                if image_description:
                    parts.append(image_description)
                if raw_description and raw_description.strip():
                    parts.append(raw_description.strip())
                
                if not parts:
                    return {
                        'item_id': int(item_id),
                        'success': False,
                        'message': '설명 정보가 필요합니다'
                    }
                
                raw_full_text = " ".join(parts).strip()
                
                # 디버깅: 임베딩에 포함될 텍스트 로그 출력 (처음 10개만)
                if int(item_id) <= 10:
                    print(f"📝 [임베딩 디버그] item_id={item_id}")
                    print(f"   - 제목: '{item_name}'")
                    print(f"   - 이미지 캡셔닝: '{image_description[:100] if image_description else '(없음)'}'")
                    print(f"   - 설명: '{raw_description[:100] if raw_description else '(없음)'}'")
                    print(f"   - 결합된 텍스트 (전처리 전): '{raw_full_text[:200]}'")
                
                final_text = preprocess_text(raw_full_text)
                if not final_text or len(final_text.strip()) == 0:
                    final_text = raw_full_text.strip()
                
                # 디버깅: 전처리 후 텍스트 (처음 10개만)
                if int(item_id) <= 10:
                    print(f"   - 전처리 후 텍스트: '{final_text[:200]}'")
                    print(f"   - 최종 임베딩 텍스트 길이: {len(final_text)}")
                
                # 3. 임베딩 벡터 생성 (저장 시이므로 is_query=False)
                embedding_vector = create_embedding_vector(final_text, is_query=False)
                
                # 4. FAISS 인덱스에 벡터 추가 (스레드 안전하게 처리)
                faiss_idx = None
                with _faiss_lock:
                    faiss_index.add(np.array([embedding_vector]))
                    faiss_idx = faiss_index.ntotal - 1
                    id_mapping[faiss_idx] = int(item_id)
                
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
        
        # 병렬 처리 (최대 10개 동시 처리)
        max_workers = min(10, len(items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                result = future.result()
                results.append(result)
                if result.get('success'):
                    successful_count += 1
        
        # FAISS 인덱스 및 매핑 정보를 디스크에 저장 (배치 완료 후 한 번만)
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
        data = request.get_json()
        raw_query = data.get('query', '')
        if not raw_query or not raw_query.strip():
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # 검색 쿼리 전처리 최소화: 원본 쿼리 우선 사용
        # BGE-M3는 한국어를 잘 처리하므로 전처리를 최소화하는 것이 성능 향상에 도움됨
        query = raw_query.strip()
        
        # 전처리는 선택적으로만 적용 (원본이 비어있을 때만)
        # 맞춤법 교정 모델이 정확한 키워드를 변경할 수 있으므로 검색 시에는 사용하지 않음
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # FAISS 인덱스 초기화 확인 (한 번만 실행)
        if not _faiss_initialized:
            initialize_faiss()
        
        # FAISS 인덱스가 비어있으면 빈 결과 반환
        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 검색어를 임베딩 벡터로 변환 (BGE-M3 사용)
        # BGE-M3는 instruction을 활용하면 검색 성능이 향상됨
        # "이 문장을 검색합니다: " 프리픽스 추가
        query_for_embedding = f"이 문장을 검색합니다: {query}"
        query_vector = create_embedding_vector(query_for_embedding, is_query=True)
        
        # 2. FAISS에서 코사인 유사도 기반 Top-K 검색
        k = min(top_k, faiss_index.ntotal)
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정 (정확도와 성능 균형)
        # 고정값 사용하여 검색 결과 일관성 유지
        if hasattr(faiss_index, 'hnsw'):
            # 고정값 사용 (검색마다 변경하지 않음)
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        
        # 검색 실행
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        debug_pairs = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1
        ]
        print(f"📈 검색 디버그: query='{query[:50]}', 결과={debug_pairs}")
        
        # 3. FAISS 인덱스 번호 → MySQL item_id 변환
        #    유사도 순서대로 정렬된 상태 유지
        item_ids = []
        scores = []
        for idx, dist in zip(indices[0], distances[0]):
            if int(idx) != -1 and int(idx) in id_mapping:
                item_ids.append(id_mapping[int(idx)])
                scores.append(float(dist))  # IndexFlatIP이므로 내적 값 (높을수록 유사)
        
        # 디버깅: 유사도 점수와 함께 출력
        result_pairs = list(zip(item_ids[:10], scores[:10]))
        print(f"🔍 자연어 검색 완료: query='{query[:30]}...', top_k={top_k}, 결과={len(item_ids)}개")
        print(f"📊 상위 10개 유사도 점수: {result_pairs}")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids,
            'scores': scores  # 유사도 점수도 함께 반환
        })
        
    except Exception as e:
        print(f"❌ 검색 실패: {str(e)}")
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
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 이미지를 임베딩 벡터로 변환
        #    AI 팀: create_embedding_from_image() 함수 구현 필요
        image_bytes = image_file.read()
        try:
            query_vector = create_embedding_from_image(image_bytes)
        except ValueError as err:
            return jsonify({'success': False, 'message': str(err)}), 400
        
        # 2. FAISS에서 유사도 검색
        k = min(top_k, faiss_index.ntotal)
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정 (고정값 사용)
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        
        # 3. FAISS 인덱스 번호 → MySQL item_id 변환
        item_ids = []
        for idx in indices[0]:
            if int(idx) in id_mapping:
                item_ids.append(id_mapping[int(idx)])
        
        print(f"🔍 이미지 검색 완료: top_k={top_k}, 결과={len(item_ids)}개")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids
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
        query = preprocess_text(raw_query)
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
        
        # 1. 기본 시맨틱 검색
        query_for_embedding = f"이 문장을 검색합니다: {query}"
        query_vector = create_embedding_vector(query_for_embedding, is_query=True)
        k = min(top_k * 3, faiss_index.ntotal)  # 더 많이 가져와서 필터링
        
        # HNSW 인덱스인 경우 ef_search 파라미터 설정 (고정값 사용)
        if hasattr(faiss_index, 'hnsw'):
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        
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

@app.route('/api/v1/embedding/delete/<int:item_id>', methods=['DELETE'])
def delete_embedding(item_id):
    """
    분실물 삭제 시 임베딩 제거
    
    프로세스:
    - FAISS에서 물리적 삭제는 하지 않음 (성능 이슈)
    - id_mapping에서만 제거하여 검색 결과에 나타나지 않도록 함
    
    Spring에서 받는 것:
    - item_id: 삭제할 분실물의 MySQL ID
    
    Spring으로 보내는 것:
    - success: 성공 여부
    """
    try:
        # 매핑에서 제거
        deleted = [k for k, v in id_mapping.items() if v == item_id]
        for k in deleted:
            del id_mapping[k]
        
        save_faiss()
        print(f"🗑️  삭제: item_id={item_id}, 제거된 벡터={len(deleted)}개")
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ 삭제 실패: {str(e)}")
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
