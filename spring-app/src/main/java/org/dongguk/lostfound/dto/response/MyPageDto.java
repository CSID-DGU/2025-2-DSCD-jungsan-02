package org.dongguk.lostfound.dto.response;

import lombok.Builder;

import java.util.List;

@Builder
public record MyPageDto(
        UserDto user,
        List<LostItemDto> myLostItems,
        Integer totalCount
) {
}

