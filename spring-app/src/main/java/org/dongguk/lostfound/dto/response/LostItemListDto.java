package org.dongguk.lostfound.dto.response;

import lombok.Builder;

import java.util.List;

@Builder
public record LostItemListDto(
        List<LostItemDto> items,
        Integer totalCount,
        Integer page,
        Integer size
) {
}

