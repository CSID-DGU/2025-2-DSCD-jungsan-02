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
    private final TmapApiService tmapApiService;

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
     * 장소별 필터링 조회 (부분 일치)
     */
    public LostItemListDto getLostItemsByLocation(String location, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "foundDate"));
        
        // 부분 일치를 위해 Specification 사용
        Specification<LostItem> spec = (root, query, cb) -> {
            if (location == null || location.trim().isEmpty()) {
                return cb.conjunction(); // 항상 true
            }
            return cb.like(
                cb.lower(root.get("location")),
                "%" + location.toLowerCase().trim() + "%"
            );
        };
        
        Page<LostItem> itemPage = lostItemRepository.findAll(spec, pageable);

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
     * 장소 필터링은 좌표 기반 반경 필터링을 사용
     */
    public LostItemListDto filterLostItems(FilterLostItemRequest request) {
        // 장소 필터링을 위한 좌표 준비
        Double filterLat = request.locationLatitude();
        Double filterLon = request.locationLongitude();
        Double filterRadius = request.locationRadius();
        
        // 장소명만 제공된 경우 좌표로 변환 (한 번만 호출)
        if ((filterLat == null || filterLon == null) && 
            request.location() != null && !request.location().trim().isEmpty()) {
            TmapApiService.TmapPlaceResult placeResult = tmapApiService.searchPlace(request.location().trim());
            if (placeResult != null) {
                filterLat = placeResult.getLatitude();
                filterLon = placeResult.getLongitude();
                log.info("장소명 '{}'을 좌표 ({}, {})로 변환하여 필터링", 
                        request.location(), filterLat, filterLon);
            }
        }
        
        // final 변수로 복사 (람다에서 사용하기 위해)
        final Double finalFilterLat = filterLat;
        final Double finalFilterLon = filterLon;
        final Double finalFilterRadius = filterRadius;
        
        // buildSpecification에 이미 변환된 좌표를 전달하여 중복 호출 방지
        Specification<LostItem> spec = buildSpecification(request, finalFilterLat, finalFilterLon, finalFilterRadius);
        Pageable pageable = PageRequest.of(request.page(), request.size(), Sort.by(Sort.Direction.DESC, "foundDate"));
        Page<LostItem> itemPage = lostItemRepository.findAll(spec, pageable);

        // 좌표 기반 필터링이 있는 경우 정확한 거리 계산으로 재필터링
        List<LostItem> filteredItems = itemPage.getContent();
        if (finalFilterLat != null && finalFilterLon != null && finalFilterRadius != null) {
            filteredItems = filteredItems.stream()
                    .filter(item -> {
                        if (item.getLatitude() == null || item.getLongitude() == null) {
                            return false;
                        }
                        double distance = calculateHaversineDistance(
                                finalFilterLat, finalFilterLon,
                                item.getLatitude(), item.getLongitude()
                        );
                        return distance <= finalFilterRadius;
                    })
                    .toList();
        }

        List<LostItemDto> items = filteredItems.stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount(items.size()) // 정확한 거리 필터링 후 개수
                .page(request.page())
                .size(request.size())
                .build();
    }
    
    /**
     * 하버사인 공식을 사용한 두 좌표 간 직선 거리 계산 (미터)
     */
    private double calculateHaversineDistance(double lat1, double lon1, double lat2, double lon2) {
        final int R = 6371000; // 지구 반경 (미터)

        double dLat = Math.toRadians(lat2 - lat1);
        double dLon = Math.toRadians(lon2 - lon1);

        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2);

        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return R * c;
    }

    /**
     * 필터 조건에 따라 Specification 생성
     * @param request 필터 요청
     * @param filterLat 이미 변환된 위도 (null이면 request에서 가져오거나 변환 시도)
     * @param filterLon 이미 변환된 경도 (null이면 request에서 가져오거나 변환 시도)
     * @param filterRadius 반경 (미터)
     */
    private Specification<LostItem> buildSpecification(FilterLostItemRequest request, 
                                                       Double filterLat, 
                                                       Double filterLon, 
                                                       Double filterRadius) {
        return (root, query, cb) -> {
            List<Predicate> predicates = new java.util.ArrayList<>();

            // 카테고리 필터
            if (request.category() != null) {
                predicates.add(cb.equal(root.get("category"), request.category()));
            }

            // 장소 필터 (좌표 기반 반경 필터링)
            // 이미 변환된 좌표가 있으면 사용, 없으면 request에서 가져오기
            Double lat = filterLat != null ? filterLat : request.locationLatitude();
            Double lon = filterLon != null ? filterLon : request.locationLongitude();
            Double radius = filterRadius != null ? filterRadius : request.locationRadius();
            
            if (lat != null && lon != null && radius != null) {
                // 좌표 기반 반경 필터링
                // 하버사인 공식으로 거리 계산하여 반경 내 아이템만 필터링
                // MySQL에서는 직접 하버사인 공식을 사용할 수 없으므로,
                // 애플리케이션 레벨에서 필터링하거나 대략적인 범위로 먼저 필터링
                // 여기서는 대략적인 위도/경도 범위로 먼저 필터링하고, 
                // 실제 거리는 애플리케이션 레벨에서 계산
                
                // 대략적인 반경 계산 (1도 ≈ 111km)
                double radiusInDegrees = radius / 111000.0;
                
                predicates.add(cb.and(
                    cb.isNotNull(root.get("latitude")),
                    cb.isNotNull(root.get("longitude")),
                    cb.between(root.get("latitude"), 
                        lat - radiusInDegrees,
                        lat + radiusInDegrees),
                    cb.between(root.get("longitude"),
                        lon - radiusInDegrees,
                        lon + radiusInDegrees)
                ));
            } else if (request.location() != null && !request.location().trim().isEmpty()) {
                // 좌표 변환이 실패했거나 좌표가 없는 경우 문자열 일치 방식으로 폴백
                log.warn("장소명 '{}'에 대한 좌표가 없어 문자열 일치 방식으로 필터링", request.location());
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

        // 장소 필터링을 위한 좌표 미리 변환 (중복 호출 방지)
        TmapApiService.TmapPlaceResult locationPlaceResult = null;
        if (request.location() != null && !request.location().trim().isEmpty()) {
            locationPlaceResult = tmapApiService.searchPlace(request.location().trim());
            if (locationPlaceResult != null) {
                log.info("검색 필터: 장소명 '{}'을 좌표 ({}, {})로 변환", 
                        request.location(), locationPlaceResult.getLatitude(), locationPlaceResult.getLongitude());
            }
        }
        
        // final 변수로 복사 (람다에서 사용하기 위해)
        final TmapApiService.TmapPlaceResult finalLocationPlaceResult = locationPlaceResult;

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
                .filter(item -> !hasFilters(request) || matchesFilters(item, request, finalLocationPlaceResult))
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
     * 장소 필터링은 좌표 기반 반경 필터링을 사용
     * @param item 분실물 아이템
     * @param request 검색 요청
     * @param locationPlaceResult 미리 변환된 장소 좌표 (null이면 request.location()으로 변환 시도)
     */
    private boolean matchesFilters(LostItem item, SearchLostItemRequest request, 
                                   TmapApiService.TmapPlaceResult locationPlaceResult) {
        // 카테고리 필터
        if (request.category() != null && !item.getCategory().equals(request.category())) {
            return false;
        }

        // 장소 필터 (좌표 기반 반경 필터링)
        if (request.location() != null && !request.location().trim().isEmpty()) {
            // 이미 변환된 좌표가 있으면 사용, 없으면 변환 시도 (폴백)
            TmapApiService.TmapPlaceResult placeResult = locationPlaceResult;
            if (placeResult == null) {
                // 폴백: 변환되지 않은 경우에만 호출 (일반적으로는 발생하지 않아야 함)
                log.warn("matchesFilters에서 장소 좌표가 전달되지 않아 변환 시도: {}", request.location());
                placeResult = tmapApiService.searchPlace(request.location().trim());
            }
            
            if (placeResult != null) {
                // 좌표 기반 반경 필터링 (기본 반경 10km)
                if (item.getLatitude() == null || item.getLongitude() == null) {
                    return false;
                }
                double distance = calculateHaversineDistance(
                        placeResult.getLatitude(), placeResult.getLongitude(),
                        item.getLatitude(), item.getLongitude()
                );
                double radius = 10000.0; // 기본값 10km
                if (distance > radius) {
                    return false;
                }
            } else {
                // 좌표 변환 실패 시 기존 문자열 일치 방식으로 폴백
                String location = request.location().toLowerCase().trim();
                if (item.getLocation() == null || !item.getLocation().toLowerCase().contains(location)) {
                    return false;
                }
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
