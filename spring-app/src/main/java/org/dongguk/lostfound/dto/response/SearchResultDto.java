package org.dongguk.lostfound.dto.response;

import lombok.Builder;

@Builder
public record SearchResultDto(
        Long itemId,
        Double distance,
        Integer rank
) {
}

