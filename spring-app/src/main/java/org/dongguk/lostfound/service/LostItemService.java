package org.dongguk.lostfound.service;

import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.Storage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.exception.GlobalErrorCode;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.type.LostItemStatus;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.dto.request.CreateLostItemRequest;
import org.dongguk.lostfound.dto.request.FilterLostItemRequest;
import org.dongguk.lostfound.dto.request.SearchLostItemRequest;
import org.dongguk.lostfound.dto.response.LostItemDto;
import org.dongguk.lostfound.dto.response.LostItemListDto;
import org.dongguk.lostfound.dto.response.StatisticsDto;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.domain.Specification;
import jakarta.persistence.criteria.Predicate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

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
    public LostItemDto createLostItem(Long userId, CreateLostItemRequest request) {
        log.info("Creating lost item: {}", request.itemName());

        // 사용자 조회
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

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
                request.latitude(),
                request.longitude(),
                request.brand(),
                imageUrl,
                null,  // embeddingId는 나중에 업데이트
                user
        );

        lostItem = lostItemRepository.save(lostItem);
        log.info("Saved lost item to MySQL with ID: {}", lostItem.getId());

        // 3. Flask AI 서버에 임베딩 생성 요청 (비동기로 처리 가능)
        try {
            flaskApiService.createEmbedding(
                    lostItem.getId(),
                    request.itemName(),  // 분실물 제목 추가
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
     * 통합 필터링 조회 (카테고리, 장소, 날짜 범위를 동시에 적용)
     */
    public LostItemListDto filterLostItems(FilterLostItemRequest request) {
        Specification<LostItem> spec = buildSpecification(request);
        Pageable pageable = PageRequest.of(request.page(), request.size(), Sort.by(Sort.Direction.DESC, "foundDate"));
        Page<LostItem> itemPage = lostItemRepository.findAll(spec, pageable);

        List<LostItemDto> items = itemPage.getContent().stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount((int) itemPage.getTotalElements())
                .page(request.page())
                .size(request.size())
                .build();
    }

    /**
     * 필터 조건에 따라 Specification 생성
     */
    private Specification<LostItem> buildSpecification(FilterLostItemRequest request) {
        return (root, query, cb) -> {
            List<Predicate> predicates = new java.util.ArrayList<>();

            // 카테고리 필터
            if (request.category() != null) {
                predicates.add(cb.equal(root.get("category"), request.category()));
            }

            // 장소 필터 (부분 일치)
            if (request.location() != null && !request.location().trim().isEmpty()) {
                predicates.add(cb.like(
                    cb.lower(root.get("location")),
                    "%" + request.location().toLowerCase().trim() + "%"
                ));
            }

            // 브랜드 필터 (부분 일치)
            if (request.brand() != null && !request.brand().trim().isEmpty()) {
                predicates.add(cb.like(
                    cb.lower(root.get("brand")),
                    "%" + request.brand().toLowerCase().trim() + "%"
                ));
            }

            // 날짜 필터 (해당 날짜 이후)
            if (request.foundDateAfter() != null) {
                predicates.add(cb.greaterThanOrEqualTo(root.get("foundDate"), request.foundDateAfter()));
            }

            return cb.and(predicates.toArray(new Predicate[0]));
        };
    }

    /**
     * AI 검색 (자연어 검색)
     * 1. Flask AI 서버에 검색어 전송
     * 2. 유사한 분실물 ID 리스트 받음
     * 3. MySQL에서 해당 분실물들 조회
     * 4. 필터가 있으면 필터 적용
     */
    public LostItemListDto searchLostItems(SearchLostItemRequest request) {
        log.info("Searching lost items with query: {}, filters: category={}, location={}, brand={}, foundDateAfter={}", 
                request.query(), request.category(), request.location(), request.brand(), request.foundDateAfter());

        // 1. Flask AI 서버에 검색 요청 (필터를 고려하여 더 많이 가져옴)
        int searchTopK = request.topK();
        if (hasFilters(request)) {
            // 필터가 있으면 더 많이 가져와서 필터링 후 상위 결과 반환
            searchTopK = request.topK() * 3;
        }
        List<Long> itemIds = flaskApiService.searchSimilarItems(
                request.query(),
                searchTopK
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
        
        // 3. Map으로 변환하여 O(1) 조회 성능 확보 (findAllById는 순서 보장 안 함)
        Map<Long, LostItem> itemMap = lostItems.stream()
                .collect(Collectors.toMap(LostItem::getId, item -> item));

        // 4. FAISS에서 반환된 순서대로 아이템 조회 및 필터 적용
        List<LostItemDto> items = itemIds.stream()
                .map(itemMap::get)
                .filter(Objects::nonNull)
                .filter(item -> !hasFilters(request) || matchesFilters(item, request))
                .map(LostItemDto::from)
                .limit(request.topK()) // 최종 결과는 요청한 개수만큼만
                .toList();

        // 디버깅: 순서 및 유사도 점수 확인 로그
        if (!items.isEmpty()) {
            log.info("✅ 최종 검색 결과 순서 (상위 5개): {}", 
                    items.stream()
                            .limit(5)
                            .map(item -> String.format("%d:%s", item.id(), item.itemName()))
                            .collect(Collectors.joining(", ")));
            log.info("📊 FAISS에서 받은 itemIds 순서 (상위 5개): {}", 
                    itemIds.stream()
                            .limit(5)
                            .map(String::valueOf)
                            .collect(Collectors.joining(", ")));
        }

        return LostItemListDto.builder()
                .items(items)
                .totalCount(items.size())
                .page(0)
                .size(items.size())
                .build();
    }

    /**
     * 필터가 있는지 확인
     */
    private boolean hasFilters(SearchLostItemRequest request) {
        return request.category() != null
                || (request.location() != null && !request.location().trim().isEmpty())
                || (request.brand() != null && !request.brand().trim().isEmpty())
                || request.foundDateAfter() != null;
    }

    /**
     * 아이템이 필터 조건에 맞는지 확인
     */
    private boolean matchesFilters(LostItem item, SearchLostItemRequest request) {
        // 카테고리 필터
        if (request.category() != null && !item.getCategory().equals(request.category())) {
            return false;
        }

        // 장소 필터 (부분 일치)
        if (request.location() != null && !request.location().trim().isEmpty()) {
            String location = request.location().toLowerCase().trim();
            if (item.getLocation() == null || !item.getLocation().toLowerCase().contains(location)) {
                return false;
            }
        }

        // 브랜드 필터 (부분 일치)
        if (request.brand() != null && !request.brand().trim().isEmpty()) {
            String brand = request.brand().toLowerCase().trim();
            if (item.getBrand() == null || !item.getBrand().toLowerCase().contains(brand)) {
                return false;
            }
        }

        // 날짜 필터 (해당 날짜 이후)
        if (request.foundDateAfter() != null) {
            if (item.getFoundDate() == null || item.getFoundDate().isBefore(request.foundDateAfter())) {
                return false;
            }
        }

        return true;
    }

    /**
     * 분실물 삭제
     */
    @Transactional
    public void deleteLostItem(Long userId, Long id) {
        LostItem lostItem = lostItemRepository.findById(id)
                .orElseThrow(() -> new CustomException(GlobalErrorCode.NOT_FOUND));

        // 본인이 등록한 분실물인지 확인
        if (!lostItem.getUser().getId().equals(userId)) {
            throw new CustomException(GlobalErrorCode.FORBIDDEN);
        }

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

    /**
     * 통계 데이터 조회
     */
    public StatisticsDto getStatistics() {
        // 전체 분실물 개수
        long totalItems = lostItemRepository.count();
        
        // 매칭된 분실물 개수 (MATCHED, COMPLETED 상태)
        long matchedItems = lostItemRepository.countByStatus(LostItemStatus.MATCHED) 
                + lostItemRepository.countByStatus(LostItemStatus.COMPLETED);
        
        // 회수 완료된 분실물 개수
        long completedItems = lostItemRepository.countByStatus(LostItemStatus.COMPLETED);
        
        // 오늘 등록된 분실물 개수
        java.time.LocalDateTime startOfDay = java.time.LocalDate.now().atStartOfDay();
        long newItemsToday = lostItemRepository.countByCreatedAtAfter(startOfDay);
        
        return StatisticsDto.builder()
                .totalItems(totalItems)
                .matchedItems(matchedItems)
                .completedItems(completedItems)
                .newItemsToday(newItemsToday)
                .build();
    }
}
