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

# ========== 전역 변수 ==========
faiss_index = None
id_mapping = {}  # FAISS 인덱스 번호 -> MySQL item_id 매핑
embedding_model: Optional[SentenceTransformer] = None
embedding_device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_faiss():
    """FAISS 인덱스 초기화 또는 로드"""
    global faiss_index, id_mapping
    
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True)
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, 'rb') as f:
            id_mapping = pickle.load(f)
        print(f"✅ FAISS 인덱스 로드: {faiss_index.ntotal}개 벡터")
    else:
        # 코사인 유사도 검색을 위해 내적 기반 인덱스 사용
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        id_mapping = {}
        print("✅ 새 FAISS 인덱스 생성")

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
    global embedding_model
    if embedding_model is None:
        print(f"📦 BGE 모델 로드: {EMBEDDING_MODEL_NAME} (device={embedding_device})")
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=embedding_device,
            trust_remote_code=True,
        )
        embedding_model.max_seq_length = 512
    return embedding_model

def describe_image_with_llava(image_bytes):
    """
    이미지에서 자연어 설명 생성 (Qwen 기반).
    """
    try:
        caption = generate_caption(image_bytes)
        return preprocess_text(caption)
    except Exception as exc:
        print(f"⚠️ 이미지 캡셔닝 실패: {exc}")
        return ""

def create_embedding_vector(text):
    """
    BGE-M3 모델을 사용하여 텍스트를 임베딩 벡터로 변환
    
    TODO: AI 팀 구현 필요
    
    Args:
        text (str): 임베딩할 텍스트 (이미지 묘사 + 사용자 입력 설명)
        
    Returns:
        numpy.ndarray: shape (EMBEDDING_DIMENSION,) 임베딩 벡터
        
    구현 가이드:
        1. BGE-M3 모델 로드 (한 번만 로드하고 재사용)
        2. 텍스트를 토크나이즈
        3. 모델에 입력하여 임베딩 벡터 추출
        4. 정규화 (normalize) 적용 (코사인 유사도 사용 시)
        5. numpy array로 반환
    """
    if not text or not text.strip():
        raise ValueError("임베딩할 텍스트가 비어 있습니다.")

    model = load_embedding_model()
    embedding = model.encode(
        [text],
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
    return create_embedding_vector(caption)


def warmup_models():
    """서버 기동 시 주요 모델을 미리 로드하여 콜드스타트를 줄임."""
    global models_warmed
    if models_warmed:
        return
    try:
        load_embedding_model()
        preprocess_text("모델 워밍업")
        models_warmed = True
        app.logger.info("✅ 모델 워밍업 완료")
    except Exception as exc:
        app.logger.warning("⚠️ 모델 워밍업 실패: %s", exc)


models_warmed = False
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
    - description: 사용자가 입력한 분실물 설명 (선택)
    - image: 분실물 이미지 파일 (선택)
    
    Spring으로 보내는 것:
    - success: 성공 여부
    - item_id: 원본 ID (확인용)
    - message: 결과 메시지
    """
    try:
        item_id = request.form.get('item_id')
        raw_description = request.form.get('description', '')
        description = preprocess_text(raw_description)
        image_file = request.files.get('image')
        
        if not item_id:
            return jsonify({'success': False, 'message': 'item_id 필요'}), 400
        
        # 1. 이미지 묘사 생성 (LLaVA 사용)
        #    AI 팀: describe_image_with_llava() 함수 구현 필요
        image_description = ""
        if image_file:
            image_bytes = image_file.read()
            image_description = describe_image_with_llava(image_bytes)
            if not image_description and raw_description:
                image_description = preprocess_text(raw_description)
            print(f"🖼️  이미지 분석 완료: {image_description[:50]}...")
        
        # 2. 이미지 묘사 + 사용자 설명 결합
        #    예) "검은색 가죽 지갑입니다. 신촌역 3번 출구에서 발견했습니다."
        parts = [p for p in [image_description, description] if p]
        if not parts and raw_description:
            parts.append(raw_description.strip())
        full_text = " ".join(parts).strip()
        
        if not full_text:
            return jsonify({'success': False, 'message': '설명 정보가 필요합니다.'}), 400
        
        print(f"🧾 임베딩 텍스트: item_id={item_id}, text='{full_text[:120]}'")
        
        # 3. 텍스트를 임베딩 벡터로 변환 (BGE-M3 사용)
        #    AI 팀: create_embedding_vector() 함수 구현 필요
        embedding_vector = create_embedding_vector(full_text)
        
        # 4. FAISS 인덱스에 벡터 추가
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
        query = preprocess_text(raw_query)
        if not query:
            query = raw_query.strip()
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        # FAISS 인덱스가 비어있으면 빈 결과 반환
        if faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 검색어를 임베딩 벡터로 변환 (BGE-M3 사용)
        #    AI 팀: create_embedding_vector() 함수 구현 필요
        query_vector = create_embedding_vector(query)
        
        # 2. FAISS에서 코사인 유사도 기반 Top-K 검색
        #    L2 거리 기반 검색 (작을수록 유사)
        #    TODO: AI 팀에서 IndexFlatIP (내적 기반)로 변경 고려 가능
        k = min(top_k, faiss_index.ntotal)
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
        for idx in indices[0]:
            if int(idx) in id_mapping:
                item_ids.append(id_mapping[int(idx)])
        
        print(f"🔍 자연어 검색 완료: query='{query[:30]}...', top_k={top_k}, 결과={len(item_ids)}개")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids
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
        
        if faiss_index.ntotal == 0:
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
        
        if faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 기본 시맨틱 검색
        query_vector = create_embedding_vector(query)
        k = min(top_k * 3, faiss_index.ntotal)  # 더 많이 가져와서 필터링
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

# Gunicorn으로 실행할 때도 FAISS 초기화
# 각 워커 프로세스가 시작될 때마다 초기화됨
initialize_faiss()

if __name__ == '__main__':
    # 개발 모드에서 직접 실행할 때만 사용
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
