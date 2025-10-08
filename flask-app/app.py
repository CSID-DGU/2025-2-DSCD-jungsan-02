from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import numpy as np
from datetime import datetime
import pickle

app = Flask(__name__)
CORS(app)

# FAISS 설정
FAISS_INDEX_PATH = 'faiss_index.idx'
FAISS_MAPPING_PATH = 'id_mapping.pkl'
EMBEDDING_DIMENSION = 768  # BGE-M3 기본 차원 (AI 팀이 수정 가능)

# 전역 변수
faiss_index = None
id_mapping = {}  # FAISS 인덱스 번호 -> MySQL item_id 매핑

def initialize_faiss():
    """FAISS 인덱스 초기화 또는 로드"""
    global faiss_index, id_mapping
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAPPING_PATH, 'rb') as f:
            id_mapping = pickle.load(f)
        print(f"✅ FAISS 인덱스 로드: {faiss_index.ntotal}개 벡터")
    else:
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        id_mapping = {}
        print("✅ 새 FAISS 인덱스 생성")

def save_faiss():
    """FAISS 스냅샷 저장"""
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAPPING_PATH, 'wb') as f:
        pickle.dump(id_mapping, f)
    print(f"💾 FAISS 저장: {faiss_index.ntotal}개 벡터")

# ========== AI 팀이 구현할 함수들 (현재는 더미) ==========

def describe_image_with_llava(image_bytes):
    """
    TODO: AI 팀 구현
    LLaVA로 이미지 묘사 생성
    """
    return "검은색 지갑입니다"  # 더미 응답

def create_embedding_vector(text):
    """
    TODO: AI 팀 구현
    BGE-M3로 텍스트를 임베딩 벡터로 변환
    """
    # 현재는 랜덤 벡터 반환
    return np.random.rand(EMBEDDING_DIMENSION).astype('float32')

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
    습득 프로세스: 이미지 → 묘사 문장 → 임베딩 → FAISS 저장
    
    Spring에서 받는 것:
    - item_id: MySQL 분실물 ID
    - description: 분실물 설명 (선택)
    - image: 이미지 파일
    
    Spring으로 보내는 것:
    - item_id: 원본 ID (확인용)
    """
    try:
        item_id = request.form.get('item_id')
        description = request.form.get('description', '')
        image_file = request.files.get('image')
        
        if not item_id:
            return jsonify({'success': False, 'message': 'item_id 필요'}), 400
        
        # 1. 이미지 묘사 생성 (AI 팀 구현 필요)
        image_description = ""
        if image_file:
            image_bytes = image_file.read()
            image_description = describe_image_with_llava(image_bytes)
        
        # 2. 묘사 + 설명 합치기
        full_text = f"{image_description} {description}".strip()
        
        # 3. 임베딩 벡터 생성 (AI 팀 구현 필요)
        embedding_vector = create_embedding_vector(full_text)
        
        # 4. FAISS에 추가
        faiss_index.add(np.array([embedding_vector]))
        faiss_idx = faiss_index.ntotal - 1
        
        # 5. 매핑 저장 (FAISS 인덱스 번호 → MySQL item_id)
        id_mapping[faiss_idx] = int(item_id)
        
        # 6. 스냅샷 저장
        save_faiss()
        
        print(f"✅ 임베딩 생성: item_id={item_id}, faiss_idx={faiss_idx}")
        
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
    검색 프로세스: 검색어 → 임베딩 → FAISS 유사도 검색 → item_id 리스트 반환
    
    Spring에서 받는 것:
    - query: 자연어 검색어
    - top_k: 반환할 개수 (기본 10)
    
    Spring으로 보내는 것:
    - item_ids: 유사한 분실물의 MySQL ID 리스트
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'success': False, 'message': '검색어 필요'}), 400
        
        if faiss_index.ntotal == 0:
            return jsonify({'success': True, 'item_ids': []})
        
        # 1. 검색어를 임베딩 벡터로 변환 (AI 팀 구현 필요)
        query_vector = create_embedding_vector(query)
        
        # 2. FAISS에서 유사도 검색
        k = min(top_k, faiss_index.ntotal)
        distances, indices = faiss_index.search(np.array([query_vector]), k)
        
        # 3. FAISS 인덱스 → MySQL item_id 변환
        item_ids = []
        for idx in indices[0]:
            if int(idx) in id_mapping:
                item_ids.append(id_mapping[int(idx)])
        
        print(f"🔍 검색 완료: query='{query}', 결과={len(item_ids)}개")
        
        return jsonify({
            'success': True,
            'item_ids': item_ids
        })
        
    except Exception as e:
        print(f"❌ 검색 실패: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/v1/embedding/delete/<int:item_id>', methods=['DELETE'])
def delete_embedding(item_id):
    """
    분실물 삭제 시 호출 (매핑만 제거, FAISS 물리 삭제는 안 함)
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

if __name__ == '__main__':
    # FAISS 초기화
    initialize_faiss()
    
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
