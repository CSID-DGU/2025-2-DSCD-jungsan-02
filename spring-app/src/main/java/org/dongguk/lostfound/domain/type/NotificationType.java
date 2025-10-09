package org.dongguk.lostfound.domain.type;

/**
 * 알림 타입
 */
public enum NotificationType {
    MATCH,      // AI 매칭 성공
    REQUEST,    // 회수 요청
    APPROVED,   // 회수 승인
    REJECTED,   // 회수 거절
    SYSTEM,     // 시스템 알림
    UPDATE      // 업데이트 알림
}

