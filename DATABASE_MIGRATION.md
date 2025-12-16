# 데이터베이스 마이그레이션 가이드

## 실행해야 할 SQL 스크립트

다음 SQL 스크립트를 MySQL 데이터베이스에 실행해주세요.

### 방법 1: SQL 파일 실행

```bash
mysql -u root -p lostfound < spring-app/src/main/resources/db/migration/add_watch_keywords_and_claim_image.sql
```

### 방법 2: MySQL 클라이언트에서 직접 실행

MySQL에 접속한 후 다음 SQL을 실행:

```sql
-- 1. watch_keywords 테이블 생성 (키워드 알림 기능)
CREATE TABLE IF NOT EXISTS watch_keywords (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME NOT NULL,
    updated_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_active (user_id, is_active),
    INDEX idx_keyword (keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. claim_requests 테이블에 image_url 컬럼 추가 (회수 요청 이미지 기능)
-- MySQL 8.0 이상에서는 IF NOT EXISTS를 지원하지 않으므로, 
-- 컬럼이 이미 존재하는지 먼저 확인하세요.
ALTER TABLE claim_requests 
ADD COLUMN image_url VARCHAR(500) NULL 
AFTER message;
```

### MySQL 8.0 미만 버전 사용 시

`IF NOT EXISTS`를 지원하지 않으므로, 다음 방법을 사용하세요:

```sql
-- 1. watch_keywords 테이블 생성
CREATE TABLE watch_keywords (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME NOT NULL,
    updated_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_active (user_id, is_active),
    INDEX idx_keyword (keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. claim_requests 테이블에 image_url 컬럼 추가
-- 컬럼이 이미 존재하는지 확인 후 실행
ALTER TABLE claim_requests 
ADD COLUMN image_url VARCHAR(500) NULL 
AFTER message;
```

## 변경 사항 요약

### 1. watch_keywords 테이블 (새로 생성)
- **목적**: 사용자가 등록한 키워드를 저장
- **주요 컬럼**:
  - `id`: 기본 키
  - `user_id`: 사용자 ID (외래 키)
  - `keyword`: 감시할 키워드
  - `is_active`: 활성화 여부
  - `created_at`, `updated_at`: 생성/수정 시간

### 2. claim_requests 테이블 수정
- **변경 사항**: `image_url` 컬럼 추가
- **목적**: 회수 요청 시 첨부한 증빙 이미지 URL 저장
- **타입**: VARCHAR(500), NULL 허용

## 확인 방법

마이그레이션 후 다음 쿼리로 확인하세요:

```sql
-- watch_keywords 테이블 확인
DESCRIBE watch_keywords;

-- claim_requests 테이블 구조 확인
DESCRIBE claim_requests;

-- image_url 컬럼이 있는지 확인
SHOW COLUMNS FROM claim_requests LIKE 'image_url';
```

## 주의사항

1. **백업**: 마이그레이션 전에 데이터베이스를 백업하세요.
2. **기존 데이터**: `claim_requests` 테이블의 기존 데이터는 그대로 유지됩니다.
3. **외래 키**: `watch_keywords.user_id`는 `users.id`를 참조합니다.


