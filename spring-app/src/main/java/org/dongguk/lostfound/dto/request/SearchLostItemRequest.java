package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;

import java.time.LocalDate;
import java.util.List;

public record SearchLostItemRequest(
        String query,
        Integer topK,
        ItemCategory category,
        String location,
        List<String> locations,  // 여러 장소 (최대 3개)
        String brand,
        LocalDate foundDateAfter,
        Double locationRadius,  // 반경 (미터, 기본값 10000m = 10km)
        Integer page,
        Integer size
) {
    public SearchLostItemRequest {
        // topK가 null이거나 0 이하면 기본값 20으로 설정 (적절한 개수)
        if (topK == null || topK <= 0) {
            topK = 20;
        }
        // 페이지네이션 기본값 설정
        if (page == null || page < 0) {
            page = 0;
        }
        if (size == null || size <= 0) {
            size = 20;
        }
        // locationRadius 기본값 설정
        if (locationRadius == null || locationRadius <= 0) {
            locationRadius = 10000.0; // 기본값 10km
        }
    }
}

