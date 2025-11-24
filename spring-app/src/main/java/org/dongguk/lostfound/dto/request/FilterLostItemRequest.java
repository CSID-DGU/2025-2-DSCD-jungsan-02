package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;

import java.time.LocalDate;

public record FilterLostItemRequest(
        ItemCategory category,
        String location,
        String brand,
        LocalDate foundDateAfter,  // 해당 날짜 이후
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
    }
}

