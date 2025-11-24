package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;

import java.time.LocalDate;

public record SearchLostItemRequest(
        String query,
        Integer topK,
        ItemCategory category,
        String location,
        String brand,
        LocalDate foundDateAfter
) {
    public SearchLostItemRequest {
        if (topK == null || topK <= 0) {
            topK = 10;
        }
    }
}

