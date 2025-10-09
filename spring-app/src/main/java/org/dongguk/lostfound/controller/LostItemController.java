package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.annotation.UserId;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.dto.request.CreateLostItemRequest;
import org.dongguk.lostfound.dto.request.SearchLostItemRequest;
import org.dongguk.lostfound.dto.response.LostItemDto;
import org.dongguk.lostfound.dto.response.LostItemListDto;
import org.dongguk.lostfound.service.LostItemService;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDate;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/lost-items")
public class LostItemController {
    private final LostItemService lostItemService;

    /**
     * 분실물 등록
     */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<LostItemDto> createLostItem(
            @UserId Long userId,
            @RequestParam String itemName,
            @RequestParam ItemCategory category,
            @RequestParam String description,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate foundDate,
            @RequestParam String location,
            @RequestParam(required = false) MultipartFile image
    ) {
        CreateLostItemRequest request = new CreateLostItemRequest(
                itemName, category, description, foundDate, location, image
        );
        
        LostItemDto result = lostItemService.createLostItem(userId, request);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물 전체 조회 (페이징)
     */
    @GetMapping
    public ResponseEntity<LostItemListDto> getAllLostItems(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        LostItemListDto result = lostItemService.getAllLostItems(page, size);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물 상세 조회
     */
    @GetMapping("/{id}")
    public ResponseEntity<LostItemDto> getLostItem(
            @PathVariable Long id
    ) {
        LostItemDto result = lostItemService.getLostItemById(id);
        return ResponseEntity.ok(result);
    }

    /**
     * 카테고리별 필터링
     */
    @GetMapping("/category/{category}")
    public ResponseEntity<LostItemListDto> getLostItemsByCategory(
            @PathVariable ItemCategory category,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        LostItemListDto result = lostItemService.getLostItemsByCategory(category, page, size);
        return ResponseEntity.ok(result);
    }

    /**
     * 날짜 범위별 필터링
     */
    @GetMapping("/date-range")
    public ResponseEntity<LostItemListDto> getLostItemsByDateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        LostItemListDto result = lostItemService.getLostItemsByDateRange(startDate, endDate, page, size);
        return ResponseEntity.ok(result);
    }

    /**
     * 장소별 필터링
     */
    @GetMapping("/location")
    public ResponseEntity<LostItemListDto> getLostItemsByLocation(
            @RequestParam String location,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        LostItemListDto result = lostItemService.getLostItemsByLocation(location, page, size);
        return ResponseEntity.ok(result);
    }

    /**
     * AI 자연어 검색
     */
    @PostMapping("/search")
    public ResponseEntity<LostItemListDto> searchLostItems(
            @RequestBody SearchLostItemRequest request
    ) {
        LostItemListDto result = lostItemService.searchLostItems(request);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물 삭제
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteLostItem(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        lostItemService.deleteLostItem(userId, id);
        return ResponseEntity.ok().build();
    }
}
