package org.dongguk.lostfound.dto.request;

public record NearbyCustodyLocationRequest(
        Double latitude,
        Double longitude,
        Integer topK
) {
}

