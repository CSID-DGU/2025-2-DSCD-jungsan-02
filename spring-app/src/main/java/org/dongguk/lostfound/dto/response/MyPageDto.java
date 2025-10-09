package org.dongguk.lostfound.dto.response;

import lombok.Builder;

import java.util.List;

@Builder
public record MyPageDto(
        UserDto user,
        List<LostItemDto> lostItems,
        Integer totalCount,
        List<ClaimRequestDto> receivedClaimRequests, // 받은 회수 요청 목록
        Integer pendingClaimCount // 대기중인 회수 요청 개수
) {
}
