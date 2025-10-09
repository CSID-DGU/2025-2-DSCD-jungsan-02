package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.type.LostItemStatus;

import java.time.LocalDate;

@Builder
public record LostItemDto(
        Long id,
        String itemName,
        ItemCategory category,
        String description,
        LocalDate foundDate,
        String location,
        String imageUrl,
        Long embeddingId,
        LostItemStatus status,
        Long userId  // 작성자 ID (권한 확인용)
) {
    public static LostItemDto from(LostItem lostItem) {
        return LostItemDto.builder()
                .id(lostItem.getId())
                .itemName(lostItem.getItemName())
                .category(lostItem.getCategory())
                .description(lostItem.getDescription())
                .foundDate(lostItem.getFoundDate())
                .location(lostItem.getLocation())
                .imageUrl(lostItem.getImageUrl())
                .embeddingId(lostItem.getEmbeddingId())
                .status(lostItem.getStatus())
                .userId(lostItem.getUser().getId())
                .build();
    }
}

