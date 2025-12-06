package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;

import java.time.LocalDate;

public record SearchLostItemRequest(
        String query,
        Integer topK,
        ItemCategory category,
        String location,
        String brand,
        LocalDate foundDateAfter,
        Integer page,
        Integer size
) {
    public SearchLostItemRequest {
        // topK가 null이거나 0 이하면 기본값 500으로 설정 (충분히 많은 결과 반환)
        if (topK == null || topK <= 0) {
            topK = 500;
        }
        // 페이지네이션 기본값 설정
        if (page == null || page < 0) {
            page = 0;
        }
        if (size == null || size <= 0) {
            size = 20;
        }
    }
}

