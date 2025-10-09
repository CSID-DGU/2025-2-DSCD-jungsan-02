package org.dongguk.lostfound.domain.type;

/**
 * 회수 요청 상태
 */
public enum ClaimStatus {
    PENDING,    // 대기중
    APPROVED,   // 승인됨
    REJECTED,   // 거절됨
    COMPLETED   // 완료됨
}

