package org.dongguk.lostfound.dto.request;

import org.dongguk.lostfound.domain.type.ItemCategory;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;

public record CreateLostItemRequest(
        String itemName,
        ItemCategory category,
        String description,
        LocalDate foundDate,
        String location,
        MultipartFile image
) {
}

