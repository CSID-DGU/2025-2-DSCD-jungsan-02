package org.dongguk.lostfound.service;

import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.Storage;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.exception.GlobalErrorCode;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.dto.request.CreateLostItemRequest;
import org.dongguk.lostfound.dto.request.SearchLostItemRequest;
import org.dongguk.lostfound.dto.response.LostItemDto;
import org.dongguk.lostfound.dto.response.LostItemListDto;
import org.dongguk.lostfound.dto.response.SearchResultDto;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDate;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class LostItemService {
    @Value("${cloud.storage.bucket}")
    private String BUCKET_NAME;
    private final Storage storage;
    private final RestClient flaskRestClient;
    private final FlaskApiService flaskApiService;
    private final UserRepository userRepository;
    private final LostItemRepository lostItemRepository;

    /**
     * 분실물 등록
     * 1. 이미지를 GCS에 업로드
     * 2. MySQL에 메타데이터 저장
     * 3. Flask AI 서버에 이미지/설명 전송하여 임베딩 생성
     */
    @Transactional
    public LostItemDto createLostItem(CreateLostItemRequest request) {
        log.info("Creating lost item: {}", request.itemName());

        // 1. 이미지 업로드 (있는 경우)
        String imageUrl = null;
        if (request.image() != null && !request.image().isEmpty()) {
            try {
                // 임시 ID로 업로드 (나중에 실제 ID로 변경 가능)
                imageUrl = uploadImage(0L, request.image().getBytes(), request.image().getOriginalFilename());
            } catch (IOException e) {
                log.error("Failed to upload image", e);
                throw new RuntimeException("이미지 업로드 실패");
            }
        }

        // 2. MySQL에 메타데이터 저장
        LostItem lostItem = LostItem.create(
                request.itemName(),
                request.category(),
                request.description(),
                request.foundDate(),
                request.location(),
                imageUrl,
                null  // embeddingId는 나중에 업데이트
        );

        lostItem = lostItemRepository.save(lostItem);
        log.info("Saved lost item to MySQL with ID: {}", lostItem.getId());

        // 3. Flask AI 서버에 임베딩 생성 요청 (비동기로 처리 가능)
        try {
            flaskApiService.createEmbedding(
                    lostItem.getId(),
                    request.description(),
                    request.image()
            );
            log.info("Embedding created for item {}", lostItem.getId());
        } catch (Exception e) {
            log.error("Failed to create embedding for item {}", lostItem.getId(), e);
            // 임베딩 생성 실패해도 분실물 등록은 성공으로 처리
        }

        return LostItemDto.from(lostItem);
    }

    /**
     * 분실물 전체 조회 (페이징)
     */
    public LostItemListDto getAllLostItems(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "id"));
        Page<LostItem> itemPage = lostItemRepository.findAll(pageable);

        List<LostItemDto> items = itemPage.getContent().stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount((int) itemPage.getTotalElements())
                .page(page)
                .size(size)
                .build();
    }

    /**
     * 분실물 상세 조회
     */
    public LostItemDto getLostItemById(Long id) {
        LostItem lostItem = lostItemRepository.findById(id)
                .orElseThrow(() -> new CustomException(GlobalErrorCode.NOT_FOUND));

        return LostItemDto.from(lostItem);
    }

    /**
     * 카테고리별 필터링 조회
     */
    public LostItemListDto getLostItemsByCategory(ItemCategory category, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "id"));
        Page<LostItem> itemPage = lostItemRepository.findByCategory(category, pageable);

        List<LostItemDto> items = itemPage.getContent().stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount((int) itemPage.getTotalElements())
                .page(page)
                .size(size)
                .build();
    }

    /**
     * 날짜 범위별 필터링 조회
     */
    public LostItemListDto getLostItemsByDateRange(LocalDate startDate, LocalDate endDate, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "foundDate"));
        Page<LostItem> itemPage = lostItemRepository.findByFoundDateBetween(startDate, endDate, pageable);

        List<LostItemDto> items = itemPage.getContent().stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount((int) itemPage.getTotalElements())
                .page(page)
                .size(size)
                .build();
    }

    /**
     * 장소별 필터링 조회
     */
    public LostItemListDto getLostItemsByLocation(String location, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "foundDate"));
        Page<LostItem> itemPage = lostItemRepository.findByLocation(location, pageable);

        List<LostItemDto> items = itemPage.getContent().stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount((int) itemPage.getTotalElements())
                .page(page)
                .size(size)
                .build();
    }

    /**
     * AI 검색 (자연어 검색)
     * 1. Flask AI 서버에 검색어 전송
     * 2. 유사한 분실물 ID 리스트 받음
     * 3. MySQL에서 해당 분실물들 조회
     */
    public LostItemListDto searchLostItems(SearchLostItemRequest request) {
        log.info("Searching lost items with query: {}", request.query());

        // 1. Flask AI 서버에 검색 요청
        List<Long> itemIds = flaskApiService.searchSimilarItems(
                request.query(),
                request.topK()
        );

        if (itemIds.isEmpty()) {
            return LostItemListDto.builder()
                    .items(List.of())
                    .totalCount(0)
                    .page(0)
                    .size(0)
                    .build();
        }

        // 2. MySQL에서 해당 분실물들 조회
        List<LostItem> lostItems = lostItemRepository.findAllById(itemIds);

        // 3. 검색 결과 순서대로 정렬 (FAISS에서 반환된 순서 유지)
        List<LostItemDto> items = itemIds.stream()
                .map(id -> lostItems.stream()
                        .filter(item -> item.getId().equals(id))
                        .findFirst()
                        .map(LostItemDto::from)
                        .orElse(null))
                .filter(item -> item != null)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount(items.size())
                .page(0)
                .size(items.size())
                .build();
    }

    /**
     * 분실물 삭제
     */
    @Transactional
    public void deleteLostItem(Long id) {
        LostItem lostItem = lostItemRepository.findById(id)
                .orElseThrow(() -> new CustomException(GlobalErrorCode.NOT_FOUND));

        // Flask AI 서버에 임베딩 삭제 요청
        try {
            flaskApiService.deleteEmbedding(id);
        } catch (Exception e) {
            log.error("Failed to delete embedding for item {}", id, e);
            // 임베딩 삭제 실패해도 계속 진행
        }

        // MySQL에서 삭제
        lostItemRepository.delete(lostItem);
        log.info("Deleted lost item with ID: {}", id);
    }

    private String uploadImage(
            Long lostItemId,
            byte[] image,
            String imageName
    ) {
        UUID uuid = UUID.randomUUID();
        String objectName = "lost" + lostItemId + "/" + imageName + uuid;

        BlobInfo blobInfo = BlobInfo.newBuilder(BUCKET_NAME, objectName)
                .setContentType(probeContentType(imageName))
                .build();
        storage.create(blobInfo, image);

        return String.format("https://storage.googleapis.com/%s/%s", BUCKET_NAME, objectName);
    }

    private String probeContentType(String name) {
        String ext = name.substring(name.lastIndexOf('.') + 1).toLowerCase();
        return switch (ext) {
            case "png" -> "image/png";
            case "jpg", "jpeg" -> "image/jpeg";
            case "gif" -> "image/gif";
            case "bmp" -> "image/bmp";
            case "webp" -> "image/webp";
            default -> "application/octet-stream";
        };
    }
}
