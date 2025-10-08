# Lost & Found API 문서

## 🔐 인증 API

### 1. 회원가입
```
POST /api/v1/auth/sign-up
Content-Type: application/json

Request:
{
  "loginId": "string",
  "password": "string"
}

Response:
{
  "accessToken": "string",
  "refreshToken": "string"
}
```

### 2. 로그인
```
POST /api/v1/auth/sign-in
Content-Type: application/json

Request:
{
  "loginId": "string",
  "password": "string"
}

Response:
{
  "accessToken": "string",
  "refreshToken": "string"
}
```

### 3. 아이디 중복 확인
```
POST /api/v1/auth/check-id
Content-Type: application/json

Request:
{
  "loginId": "string"
}

Response:
{
  "success": true,
  "message": "사용 가능한 아이디입니다."
}
```

---

## 🔍 분실물 API

### 1. 분실물 등록
```
POST /api/v1/lost-items
Content-Type: multipart/form-data
Authorization: Bearer {accessToken}

Request:
- itemName: string (필수)
- category: ELECTRONICS | CLOTHING | ACCESSORIES | DOCUMENTS | BOOKS | KEYS | WALLET | OTHER
- description: string (필수)
- foundDate: yyyy-MM-dd (필수)
- location: string (필수)
- image: file (선택)

Response:
{
  "success": true,
  "data": {
    "id": 1,
    "itemName": "검은색 지갑",
    "category": "WALLET",
    "description": "검은색 가죽 지갑입니다",
    "foundDate": "2025-10-08",
    "location": "신촌역 3번 출구",
    "imageUrl": "https://storage.googleapis.com/...",
    "embeddingId": null
  }
}
```

### 2. 분실물 전체 조회 (페이징)
```
GET /api/v1/lost-items?page=0&size=20

Response:
{
  "success": true,
  "data": {
    "items": [ LostItemDto... ],
    "totalCount": 100,
    "page": 0,
    "size": 20
  }
}
```

### 3. 분실물 상세 조회
```
GET /api/v1/lost-items/{id}

Response:
{
  "success": true,
  "data": {
    "id": 1,
    "itemName": "검은색 지갑",
    ...
  }
}
```

### 4. 카테고리별 필터링
```
GET /api/v1/lost-items/category/{category}?page=0&size=20

Example: GET /api/v1/lost-items/category/WALLET?page=0&size=20
```

### 5. 날짜 범위별 필터링
```
GET /api/v1/lost-items/date-range?startDate=2025-10-01&endDate=2025-10-08&page=0&size=20
```

### 6. 장소별 필터링
```
GET /api/v1/lost-items/location?location=신촌역&page=0&size=20
```

### 7. AI 자연어 검색 ⭐
```
POST /api/v1/lost-items/search
Content-Type: application/json

Request:
{
  "query": "검은색 지갑을 찾습니다",
  "topK": 10
}

Response:
{
  "success": true,
  "data": {
    "items": [ LostItemDto... ],
    "totalCount": 5,
    "page": 0,
    "size": 5
  }
}
```

### 8. 분실물 삭제
```
DELETE /api/v1/lost-items/{id}
Authorization: Bearer {accessToken}

Response:
{
  "success": true,
  "data": null
}
```

---

## 👤 사용자 API

### 1. 내 정보 조회
```
GET /api/v1/users/me
Authorization: Bearer {accessToken}

Response:
{
  "success": true,
  "data": {
    "id": 1,
    "loginId": "user123"
  }
}
```

### 2. 마이페이지 조회
```
GET /api/v1/users/mypage
Authorization: Bearer {accessToken}

Response:
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "loginId": "user123"
    },
    "myLostItems": [],
    "totalCount": 0
  }
}
```

---

## 🤖 Flask AI 서버 API (내부 통신용)

### 1. 임베딩 생성
```
POST /api/v1/embedding/create
Content-Type: multipart/form-data

Request:
- item_id: string (필수)
- description: string
- image: file

Response:
{
  "success": true,
  "item_id": 1,
  "message": "임베딩 생성 완료"
}
```

### 2. 유사도 검색
```
POST /api/v1/embedding/search
Content-Type: application/json

Request:
{
  "query": "검은색 지갑",
  "top_k": 10
}

Response:
{
  "success": true,
  "item_ids": [1, 5, 3, 8, 12]
}
```

### 3. 임베딩 삭제
```
DELETE /api/v1/embedding/delete/{item_id}

Response:
{
  "success": true
}
```

---

## 📦 데이터 구조

### ItemCategory (분실물 카테고리)
- `ELECTRONICS`: 전자기기
- `CLOTHING`: 의류
- `ACCESSORIES`: 악세서리
- `DOCUMENTS`: 서류/증명서
- `BOOKS`: 책/노트
- `KEYS`: 열쇠
- `WALLET`: 지갑
- `OTHER`: 기타

---

## 🔄 시스템 플로우

### 분실물 등록 프로세스
1. 사용자가 Spring Boot에 분실물 정보 + 이미지 전송
2. Spring Boot가 이미지를 GCS에 업로드
3. Spring Boot가 메타데이터를 MySQL에 저장
4. Spring Boot가 Flask AI 서버에 이미지 + 설명 전송
5. Flask AI 서버가 LLaVA로 이미지 묘사 생성
6. Flask AI 서버가 BGE-M3로 임베딩 벡터 생성
7. Flask AI 서버가 FAISS에 벡터 저장 및 스냅샷 저장
8. Spring Boot가 사용자에게 응답 반환

### AI 검색 프로세스
1. 사용자가 Spring Boot에 자연어 검색어 전송
2. Spring Boot가 Flask AI 서버에 검색어 전송
3. Flask AI 서버가 BGE-M3로 검색어 임베딩
4. Flask AI 서버가 FAISS에서 유사도 검색
5. Flask AI 서버가 Top K 분실물 ID 리스트 반환
6. Spring Boot가 MySQL에서 해당 분실물들 조회
7. Spring Boot가 사용자에게 결과 반환

---

## 🚀 AI 팀 작업 사항

Flask 앱의 `app.py`에서 다음 두 함수만 구현하면 됩니다:

```python
def describe_image_with_llava(image_bytes):
    """
    LLaVA로 이미지 묘사 생성
    
    Args:
        image_bytes: 이미지 바이트
    
    Returns:
        str: 이미지 묘사 문장
    """
    # TODO: LLaVA 모델 구현
    pass

def create_embedding_vector(text):
    """
    BGE-M3로 텍스트를 임베딩 벡터로 변환
    
    Args:
        text: 텍스트 (이미지 묘사 + 분실물 설명)
    
    Returns:
        numpy.ndarray: shape (768,) 임베딩 벡터
    """
    # TODO: BGE-M3 모델 구현
    pass
```

현재는 더미 함수로 구현되어 있으며, FAISS 저장/검색 로직은 이미 완성되어 있습니다.

