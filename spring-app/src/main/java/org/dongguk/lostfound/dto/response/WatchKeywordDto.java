package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.watchkeyword.WatchKeyword;

import java.time.LocalDateTime;

@Builder
public record WatchKeywordDto(
        Long id,
        String keyword,
        Boolean isActive,
        LocalDateTime createdAt
) {
    public static WatchKeywordDto from(WatchKeyword watchKeyword) {
        return WatchKeywordDto.builder()
                .id(watchKeyword.getId())
                .keyword(watchKeyword.getKeyword())
                .isActive(watchKeyword.getIsActive())
                .createdAt(watchKeyword.getCreatedAt())
                .build();
    }
}


