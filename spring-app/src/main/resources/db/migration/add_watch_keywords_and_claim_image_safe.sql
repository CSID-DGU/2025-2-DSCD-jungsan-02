-- ============================================
-- 안전한 데이터베이스 마이그레이션 스크립트
-- (컬럼 존재 여부 확인 후 실행)
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

-- 2. claim_requests 테이블에 image_url 컬럼 추가 (안전한 방법)
-- 컬럼이 존재하지 않을 때만 추가
SET @dbname = DATABASE();
SET @tablename = 'claim_requests';
SET @columnname = 'image_url';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (TABLE_SCHEMA = @dbname)
      AND (TABLE_NAME = @tablename)
      AND (COLUMN_NAME = @columnname)
  ) > 0,
  'SELECT 1', -- 컬럼이 이미 존재하면 아무것도 하지 않음
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' VARCHAR(500) NULL AFTER message')
));
PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- ============================================
-- 마이그레이션 완료
-- ============================================


