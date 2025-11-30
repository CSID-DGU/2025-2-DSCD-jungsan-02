package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;

import java.time.LocalDate;

public record FilterLostItemRequest(
        ItemCategory category,
        String location,
        String brand,
        LocalDate foundDateAfter,  // 해당 날짜 이후
        Double locationLatitude,   // 장소 필터링용 좌표 (위도)
        Double locationLongitude,  // 장소 필터링용 좌표 (경도)
        Double locationRadius,      // 반경 (미터, 기본값 10000m = 10km)
        Integer page,
        Integer size
) {
    public FilterLostItemRequest {
        if (page == null || page < 0) {
            page = 0;
        }
        if (size == null || size <= 0) {
            size = 20;
        }
        if (locationRadius == null || locationRadius <= 0) {
            locationRadius = 10000.0; // 기본값 10km
        }
    }
}

