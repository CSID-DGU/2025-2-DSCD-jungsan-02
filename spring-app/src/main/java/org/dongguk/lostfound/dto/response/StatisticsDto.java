package org.dongguk.lostfound.dto.response;

import lombok.Builder;

@Builder
public record StatisticsDto(
        Long totalItems,      // 전체 등록된 분실물 개수
        Long matchedItems,    // 매칭된 분실물 개수
        Long completedItems,  // 회수 완료된 분실물 개수
        Long newItemsToday    // 오늘 등록된 분실물 개수
) {
}

