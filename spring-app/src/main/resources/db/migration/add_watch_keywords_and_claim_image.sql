-- ============================================
-- 데이터베이스 마이그레이션 스크립트
-- 실행 날짜: 2025-01-XX
-- ============================================

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
-- 주의: 컬럼이 이미 존재하면 에러가 발생할 수 있습니다.
-- 에러 발생 시 "Duplicate column name 'image_url'" 메시지가 나오는데, 이는 정상입니다.
ALTER TABLE claim_requests 
ADD COLUMN image_url VARCHAR(500) NULL 
AFTER message;

-- ============================================
-- 마이그레이션 완료
-- ============================================

