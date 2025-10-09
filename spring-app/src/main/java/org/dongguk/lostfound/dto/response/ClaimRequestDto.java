package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.claim.ClaimRequest;
import org.dongguk.lostfound.domain.type.ClaimStatus;

import java.time.LocalDateTime;

@Builder
public record ClaimRequestDto(
        Long id,
        Long lostItemId,
        String lostItemName,
        Long claimerId,
        String claimerLoginId,
        ClaimStatus status,
        String message,
        LocalDateTime createdAt
) {
    public static ClaimRequestDto from(ClaimRequest claimRequest) {
        return ClaimRequestDto.builder()
                .id(claimRequest.getId())
                .lostItemId(claimRequest.getLostItem().getId())
                .lostItemName(claimRequest.getLostItem().getItemName())
                .claimerId(claimRequest.getClaimer().getId())
                .claimerLoginId(claimRequest.getClaimer().getLoginId())
                .status(claimRequest.getStatus())
                .message(claimRequest.getMessage())
                .createdAt(claimRequest.getCreatedAt())
                .build();
    }
}

