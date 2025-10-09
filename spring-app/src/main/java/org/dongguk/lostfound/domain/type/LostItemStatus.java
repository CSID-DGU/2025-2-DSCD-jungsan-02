package org.dongguk.lostfound.domain.type;

/**
 * 분실물 상태
 * - REGISTERED: 등록됨 (초기 상태)
 * - MATCHED: 매칭중 (분실자가 회수 요청을 보낸 상태)
 * - COMPLETED: 회수완료 (습득자가 회수 승인한 상태)
 * - EXPIRED: 만료됨 (오래된 게시물)
 */
public enum LostItemStatus {
    REGISTERED,   // 등록됨
    MATCHED,      // 매칭중
    COMPLETED,    // 회수완료
    EXPIRED       // 만료됨
}

