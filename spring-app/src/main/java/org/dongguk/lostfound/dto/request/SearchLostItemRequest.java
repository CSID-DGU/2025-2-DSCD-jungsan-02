package org.dongguk.lostfound.dto.request;

public record SearchLostItemRequest(
        String query,
        Integer topK
) {
    public SearchLostItemRequest {
        if (topK == null || topK <= 0) {
            topK = 10;
        }
    }
}

