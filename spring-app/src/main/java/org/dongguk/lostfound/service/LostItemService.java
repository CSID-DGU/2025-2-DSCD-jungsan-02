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
import java.util.Set;
import java.util.concurrent.CompletableFuture;
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
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "foundDate"));
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
                // 반경이 지정되지 않았으면 기본 반경 10km 사용
                if (filterRadius == null) {
                    filterRadius = 10000.0; // 10km
                    log.info("장소명 '{}'을 좌표 ({}, {})로 변환하여 필터링 (기본 반경: 10km)", 
                            request.location(), filterLat, filterLon);
                } else {
                    log.info("장소명 '{}'을 좌표 ({}, {})로 변환하여 필터링 (반경: {}m)", 
                            request.location(), filterLat, filterLon, filterRadius);
                }
            } else {
                // TMap API 쿼터 초과 여부 확인
                if (tmapApiService.isQuotaExceeded()) {
                    log.error("⚠️ TMap API 쿼터 초과로 인해 장소명 '{}'의 좌표 변환이 실패했습니다. 문자열 일치 방식으로 필터링합니다.", 
                            request.location());
                } else {
                    log.warn("⚠️ 장소명 '{}'에 대한 좌표를 찾을 수 없습니다. 문자열 일치 방식으로 필터링합니다.", 
                            request.location());
                }
            }
        }
        
        // final 변수로 복사 (람다에서 사용하기 위해)
        final Double finalFilterLat = filterLat;
        final Double finalFilterLon = filterLon;
        final Double finalFilterRadius = filterRadius;
        
        // buildSpecification에 이미 변환된 좌표를 전달하여 중복 호출 방지
        Specification<LostItem> spec = buildSpecification(request, finalFilterLat, finalFilterLon, finalFilterRadius);
        
        // 페이징 전에 전체 데이터를 가져와서 필터링 (일관성 유지)
        // 좌표 기반 필터링이 있는 경우 정확한 거리 계산을 위해, 없는 경우에도 일관성을 위해 전체 데이터 처리
        List<LostItem> allCandidates = lostItemRepository.findAll(spec);
        
        // 좌표 기반 필터링이 있는 경우: 정확한 거리 계산으로 재필터링
        List<LostItem> filteredItems;
        if (finalFilterLat != null && finalFilterLon != null && finalFilterRadius != null) {
            filteredItems = allCandidates.stream()
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
                    .sorted((a, b) -> b.getFoundDate().compareTo(a.getFoundDate())) // 최신순 정렬
                    .toList();
        } else {
            // 좌표 기반 필터링이 없는 경우: 이미 buildSpecification에서 필터링 완료, 정렬만 적용
            filteredItems = allCandidates.stream()
                    .sorted((a, b) -> b.getFoundDate().compareTo(a.getFoundDate())) // 최신순 정렬
                    .toList();
        }
        
        // 필터링 후 전체 개수
        int totalCount = filteredItems.size();
        
        // 메모리에서 페이징 적용
        int page = request.page();
        int size = request.size();
        int start = page * size;
        int end = Math.min(start + size, totalCount);
        
        List<LostItem> pagedItems = start < totalCount ? filteredItems.subList(start, end) : List.of();
        
        List<LostItemDto> items = pagedItems.stream()
                .map(LostItemDto::from)
                .toList();

        return LostItemListDto.builder()
                .items(items)
                .totalCount(totalCount) // 필터링 후 전체 개수
                .page(page)
                .size(size)
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
            
            // 좌표는 있는데 반경이 없으면 기본 반경 10km 사용
            if (lat != null && lon != null && radius == null && 
                request.location() != null && !request.location().trim().isEmpty()) {
                radius = 10000.0; // 10km
                log.debug("반경이 지정되지 않아 기본 반경 10km 사용: 장소명={}", request.location());
            }
            
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
                // 더 넓은 범위로 검색하기 위해 키워드를 분리하여 검색
                String locationKeyword = request.location().toLowerCase().trim();
                log.warn("장소명 '{}'에 대한 좌표가 없어 문자열 일치 방식으로 필터링", locationKeyword);
                
                // 키워드가 '역', '구', '동' 등으로 끝나는 경우 이를 제거하고 더 넓게 검색
                // 예: '강남역' -> '강남' 또는 '강남역' 모두 검색
                String locationKeywordWithoutSuffix = locationKeyword;
                if (locationKeyword.endsWith("역")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                } else if (locationKeyword.endsWith("구")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                } else if (locationKeyword.endsWith("동")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                }
                
                // 원본 키워드와 접미사 제거한 키워드 모두 검색 (OR 조건)
                if (!locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                    predicates.add(cb.or(
                        cb.like(cb.lower(root.get("location")), "%" + locationKeyword + "%"),
                        cb.like(cb.lower(root.get("location")), "%" + locationKeywordWithoutSuffix + "%")
                    ));
                } else {
                    predicates.add(cb.like(
                        cb.lower(root.get("location")),
                        "%" + locationKeyword + "%"
                    ));
                }
            }

            // 브랜드 필터 (부분 일치)
            // itemName, description, brand 필드 모두에서 검색
            if (request.brand() != null && !request.brand().trim().isEmpty()) {
                String brandKeyword = request.brand().toLowerCase().trim();
                // JPA의 like는 특수문자 이스케이프 처리를 위해 이스케이프 문자를 명시
                // '%'와 '_'를 리터럴로 사용하기 위해 이스케이프 문자 '!' 사용
                char escapeChar = '!';
                // 사용자 입력에 특수문자가 있으면 이스케이프 처리
                String escapedKeyword = brandKeyword
                    .replace("!", "!!")  // 이스케이프 문자 자체를 이스케이프
                    .replace("%", "!%")  // %를 리터럴로
                    .replace("_", "!_"); // _를 리터럴로
                String pattern = "%" + escapedKeyword + "%";
                
                // itemName, description, brand 중 하나라도 매치되면 통과 (OR 조건)
                predicates.add(cb.or(
                    cb.like(cb.lower(root.get("itemName")), pattern, escapeChar),
                    cb.like(cb.lower(root.get("description")), pattern, escapeChar),
                    cb.and(
                        cb.isNotNull(root.get("brand")),
                        cb.like(cb.lower(root.get("brand")), pattern, escapeChar)
                    )
                ));
                log.info("브랜드 필터 적용: '{}' -> itemName, description, brand 필드에서 검색", brandKeyword);
            }

            // 날짜 필터 (해당 날짜 이후)
            if (request.foundDateAfter() != null) {
                predicates.add(cb.greaterThanOrEqualTo(root.get("foundDate"), request.foundDateAfter()));
            }

            return cb.and(predicates.toArray(new Predicate[0]));
        };
    }

    /**
     * AI 검색 (하이브리드 검색: 키워드 매칭 + 시맨틱 검색)
     * 1. 키워드 기반 검색 (제목/설명/브랜드에 검색어 포함) - 우선순위 높음
     * 2. Flask AI 서버에 검색어 전송하여 시맨틱 검색
     * 3. 키워드 매칭 결과를 상위에 배치하고 시맨틱 검색 결과 추가
     * 4. 필터가 있으면 필터 적용
     */
    public LostItemListDto searchLostItems(SearchLostItemRequest request) {
        log.debug("Searching lost items with query: {}, filters: category={}, location={}, brand={}, foundDateAfter={}", 
                request.query(), request.category(), request.location(), request.brand(), request.foundDateAfter());

        String searchQuery = request.query() != null ? request.query().trim() : "";
        if (searchQuery.isEmpty()) {
            log.warn("검색어가 비어있습니다.");
            return LostItemListDto.builder()
                    .items(List.of())
                    .totalCount(0)
                    .page(0)
                    .size(0)
                    .build();
        }

        // 장소 필터링을 위한 좌표 미리 변환 (중복 호출 방지)
        // 여러 장소가 있으면 각각 변환, 없으면 단일 장소 변환
        List<TmapApiService.TmapPlaceResult> locationPlaceResults = new java.util.ArrayList<>();
        TmapApiService.TmapPlaceResult locationPlaceResult = null;
        Double locationRadius = request.locationRadius();
        
        // 여러 장소가 있는 경우
        if (request.locations() != null && !request.locations().isEmpty()) {
            for (String loc : request.locations()) {
                if (loc != null && !loc.trim().isEmpty()) {
                    TmapApiService.TmapPlaceResult placeResult = tmapApiService.searchPlace(loc.trim());
                    if (placeResult != null) {
                        locationPlaceResults.add(placeResult);
                        log.info("검색 필터: 장소명 '{}'을 좌표 ({}, {})로 변환", 
                                loc, placeResult.getLatitude(), placeResult.getLongitude());
                    }
                }
            }
        } 
        // 단일 장소가 있는 경우
        else if (request.location() != null && !request.location().trim().isEmpty()) {
            locationPlaceResult = tmapApiService.searchPlace(request.location().trim());
            if (locationPlaceResult != null) {
                locationPlaceResults.add(locationPlaceResult);
                log.info("검색 필터: 장소명 '{}'을 좌표 ({}, {})로 변환", 
                        request.location(), locationPlaceResult.getLatitude(), locationPlaceResult.getLongitude());
            }
        }
        
        // final 변수로 복사 (람다에서 사용하기 위해)
        final List<TmapApiService.TmapPlaceResult> finalLocationPlaceResults = locationPlaceResults;
        final TmapApiService.TmapPlaceResult finalLocationPlaceResult = 
                locationPlaceResults.isEmpty() ? null : locationPlaceResults.get(0); // 하위 호환성
        final Double finalLocationRadius = locationRadius;

        // 1. 키워드 검색과 시맨틱 검색을 병렬로 실행 (성능 개선)
        // 시맨틱 검색 topK 최적화: 키워드 매칭 결과를 고려하여 필요한 만큼만 가져옴
        // 키워드 매칭이 있으면 그만큼 덜 가져와도 됨
        // 실제 필요한 개수만 요청
        int requestedTopK = request.topK();
        // 필터가 있으면 더 많이 가져와야 하지만, 없으면 정확히 필요한 만큼만
        boolean hasFilters = hasFilters(request);
        int semanticSearchTopK;
        if (hasFilters) {
            // 필터가 있으면 필터링 후에도 충분한 결과를 얻기 위해 여유 있게
            semanticSearchTopK = Math.min(
                    Math.max(requestedTopK * 2, 20),  // 최소 20개, 최대 topK * 2
                    100  // 최대 100개로 제한
            );
        } else {
            // 필터가 없으면 정확히 필요한 만큼만 (여유 없이)
            semanticSearchTopK = requestedTopK;  // 정확히 요청한 만큼만
        }
        log.info("시맨틱 검색 topK 설정: requestedTopK={}, semanticSearchTopK={}, hasFilters={}", 
                requestedTopK, semanticSearchTopK, hasFilters);
        
        // 병렬 실행을 위한 CompletableFuture 사용
        CompletableFuture<List<LostItem>> keywordSearchFuture = 
                CompletableFuture.supplyAsync(() -> {
                    try {
                        log.debug("키워드 검색 시작: query='{}'", searchQuery);
                        List<LostItem> result = searchByKeyword(searchQuery, request, finalLocationRadius);
                        log.info("키워드 검색 완료: {}개 결과", result.size());
                        return result;
                    } catch (Exception e) {
                        log.error("키워드 검색 실패: {}", e.getMessage(), e);
                        return List.<LostItem>of(); // 빈 리스트 반환
                    }
                });
        
        CompletableFuture<FlaskApiService.SearchResult> semanticSearchFuture = 
                CompletableFuture.supplyAsync(() -> {
                    try {
                        log.info("🔍 시맨틱 검색 시작: query='{}', topK={}", searchQuery, semanticSearchTopK);
                        
                        // 1단계: 넓게 후보 회수 (메타데이터 없이)
                        int recallK = Math.min(semanticSearchTopK * 5, 500); // 넓게 후보 수집
                        FlaskApiService.SearchResult recallResult = flaskApiService.searchSimilarItemsWithScores(searchQuery, recallK);
                        List<Long> candidateIds = recallResult.getItemIds();
                        
                        if (candidateIds.isEmpty()) {
                            log.warn("⚠️ 후보 회수 결과가 비어있습니다. query: '{}'", searchQuery);
                            return new FlaskApiService.SearchResult(List.of(), List.of());
                        }
                        
                        log.info("후보 회수 완료: {}개 후보", candidateIds.size());
                        
                        // 2단계: 후보들의 메타데이터 조회
                        Map<Long, FlaskApiService.ItemMetadata> metadataMap = new java.util.HashMap<>();
                        int batchSize = 500;
                        for (int i = 0; i < candidateIds.size(); i += batchSize) {
                            int end = Math.min(i + batchSize, candidateIds.size());
                            List<Long> batchIds = candidateIds.subList(i, end);
                            List<LostItem> batchItems = lostItemRepository.findAllById(batchIds);
                            for (LostItem item : batchItems) {
                                metadataMap.put(item.getId(), new FlaskApiService.ItemMetadata(
                                    item.getCategory().name(),
                                    item.getDescription(),
                                    item.getItemName(),
                                    item.getBrand() != null ? item.getBrand() : ""
                                ));
                            }
                        }
                        
                        log.info("메타데이터 조회 완료: {}개 항목", metadataMap.size());
                        
                        // 3단계: 메타데이터와 함께 게이팅 및 재정렬 수행
                        FlaskApiService.SearchResult finalResult = flaskApiService.searchSimilarItemsWithScores(
                            searchQuery, semanticSearchTopK, metadataMap);
                        
                        log.info("✅ 시맨틱 검색 완료: {}개 결과 (query: '{}')", finalResult.getItemIds().size(), searchQuery);
                        if (finalResult.getItemIds().isEmpty()) {
                            log.warn("⚠️ 최종 검색 결과가 비어있습니다. query: '{}', topK: {}", searchQuery, semanticSearchTopK);
                        }
                        return finalResult;
                    } catch (Exception e) {
                        log.error("❌ 시맨틱 검색 실패: query='{}', 에러: {}", searchQuery, e.getMessage(), e);
                        e.printStackTrace();
                        return new FlaskApiService.SearchResult(List.of(), List.of()); // 빈 결과 반환
                    }
                });
        
        // 두 검색 결과를 모두 기다림 (예외 처리 포함)
        List<LostItem> keywordMatchedItems;
        FlaskApiService.SearchResult searchResult;
        try {
            keywordMatchedItems = keywordSearchFuture.join();
            searchResult = semanticSearchFuture.join();
        } catch (Exception e) {
            log.error("검색 실행 중 예외 발생: {}", e.getMessage(), e);
            // 예외 발생 시 빈 결과 반환
            keywordMatchedItems = List.of();
            searchResult = new FlaskApiService.SearchResult(List.of(), List.of());
        }
        
        Set<Long> keywordMatchedIds = keywordMatchedItems.stream()
                .map(LostItem::getId)
                .collect(Collectors.toSet());
        
        List<Long> semanticItemIds = searchResult.getItemIds();
        List<Double> semanticScores = searchResult.getScores();
        
        log.info("키워드 매칭 결과: {}개, 시맨틱 검색 결과: {}개 (검색어: '{}')", 
                keywordMatchedItems.size(), semanticItemIds.size(), searchQuery);
        
        // 검색 결과가 비어있을 때 경고 로그 출력
        if (keywordMatchedItems.isEmpty() && semanticItemIds.isEmpty()) {
            log.warn("⚠️ 검색 결과가 비어있습니다. 검색어: '{}', 필터: location={}, locations={}, locationRadius={}, category={}, brand={}", 
                    searchQuery, request.location(), request.locations(), request.locationRadius(), 
                    request.category(), request.brand());
        }

        // 2. 점수 기반 필터링을 먼저 수행하여 불필요한 DB 조회 방지
        double scoreThreshold = 0.3;
        List<Long> filteredSemanticIds = new java.util.ArrayList<>();
        Map<Long, Double> itemScoreMap = new java.util.HashMap<>();
        
        for (int i = 0; i < semanticItemIds.size(); i++) {
            Long itemId = semanticItemIds.get(i);
            // 키워드 매칭 결과는 제외
            if (keywordMatchedIds.contains(itemId)) {
                continue;
            }
            
            Double score = i < semanticScores.size() ? semanticScores.get(i) : 0.0;
            // 점수 필터링: 임계값 이상인 것만 포함
            if (score >= scoreThreshold) {
                filteredSemanticIds.add(itemId);
                itemScoreMap.put(itemId, score);
            }
        }
        
        log.debug("시맨틱 검색 점수 필터링: 전체 {}개 중 점수 {} 이상인 아이템 {}개", 
                semanticItemIds.size(), scoreThreshold, filteredSemanticIds.size());
        
        // 3. 필터링된 ID만 DB에서 조회 (배치로 나누어 조회)
        List<LostItem> scoredSemanticItems = new java.util.ArrayList<>();
        if (!filteredSemanticIds.isEmpty()) {
            // 배치 크기: MySQL IN 절 제한 고려 (일반적으로 1000개 이하 권장)
            int batchSize = 500;
            Map<Long, LostItem> itemMap = new java.util.HashMap<>();
            for (int i = 0; i < filteredSemanticIds.size(); i += batchSize) {
                int end = Math.min(i + batchSize, filteredSemanticIds.size());
                List<Long> batchIds = filteredSemanticIds.subList(i, end);
                List<LostItem> batchItems = lostItemRepository.findAllById(batchIds);
                // Map에 저장하여 ID로 빠르게 조회 가능하게 함
                for (LostItem item : batchItems) {
                    itemMap.put(item.getId(), item);
                }
            }
            
            // FAISS에서 반환된 순서(유사도 점수 순서)대로 정렬하여 추가
            // filteredSemanticIds는 이미 유사도 점수 순서대로 정렬되어 있음
            for (Long itemId : filteredSemanticIds) {
                LostItem item = itemMap.get(itemId);
                if (item != null) {
                    scoredSemanticItems.add(item);
                }
            }
        }

        // 6. 키워드 매칭 결과와 시맨틱 검색 결과 합치기 (키워드 매칭이 우선)
        List<LostItem> combinedItems = new java.util.ArrayList<>();
        combinedItems.addAll(keywordMatchedItems); // 키워드 매칭 결과를 먼저 추가
        combinedItems.addAll(scoredSemanticItems);  // 시맨틱 검색 결과 추가 (유사도 점수 순서 유지)
        
        // 7. 필터 적용 (hasFilters는 이미 510번 줄에서 선언됨)
        log.info("필터 적용 여부: {}, 필터 조건: category={}, location={}, locations={}, locationRadius={}, brand={}, foundDateAfter={}", 
                hasFilters, request.category(), request.location(), request.locations(), request.locationRadius(), request.brand(), request.foundDateAfter());
        
        List<LostItem> filteredItems;
        if (!hasFilters) {
            // 필터가 없으면 모두 통과
            filteredItems = combinedItems;
        } else {
            // 필터가 있으면 필터링 적용
            filteredItems = combinedItems.stream()
                    .filter(item -> {
                        boolean matches = matchesFilters(item, request, finalLocationPlaceResults, finalLocationRadius);
                        if (!matches) {
                            log.debug("아이템 필터링 제외: itemId={}, itemName={}, category={}, brand={}, location={}", 
                                    item.getId(), item.getItemName(), item.getCategory(), item.getBrand(), item.getLocation());
                        }
                        return matches;
                    })
                    .toList();
        }
        
        log.info("필터 적용 전: {}개 (키워드: {}, 시맨틱: {}), 필터 적용 후: {}개", 
                combinedItems.size(), keywordMatchedItems.size(), scoredSemanticItems.size(), filteredItems.size());
        
        // 필터 적용 후 결과가 비어있을 때 경고 로그 출력
        if (combinedItems.size() > 0 && filteredItems.isEmpty()) {
            log.warn("⚠️ 필터 적용 후 모든 결과가 제외되었습니다. 필터 조건: location={}, locations={}, locationRadius={}, category={}, brand={}, foundDateAfter={}", 
                    request.location(), request.locations(), request.locationRadius(), 
                    request.category(), request.brand(), request.foundDateAfter());
        }
        
        // 8. 페이지네이션 적용
        int totalCount = filteredItems.size();
        int page = request.page() != null ? request.page() : 0;
        int size = request.size() != null ? request.size() : 20;
        
        // 메모리에서 페이징 적용
        int start = page * size;
        int end = Math.min(start + size, totalCount);
        
        List<LostItem> pagedItems = start < totalCount ? filteredItems.subList(start, end) : List.of();
        
        // 9. 최종 결과 반환 (키워드 매칭 우선, 그 다음 시맨틱 검색)
        List<LostItemDto> items = pagedItems.stream()
                .map(LostItemDto::from)
                .toList();
        
        // 요약 로그만 출력 (상세 로그는 debug 레벨로)
        if (!items.isEmpty()) {
            // 카테고리별 분포 확인 (전체 결과 기준)
            Map<String, Long> categoryCounts = filteredItems.stream()
                    .collect(Collectors.groupingBy(
                            item -> item.getCategory() != null ? item.getCategory().toString() : "NULL",
                            Collectors.counting()
                    ));
            
            // 키워드 매칭 vs 시맨틱 검색 비율 (전체 결과 기준)
            long keywordCount = filteredItems.stream()
                    .filter(item -> keywordMatchedIds.contains(item.getId()))
                    .count();
            
            log.info("검색 완료: 전체 {}개 (키워드: {}개, 시맨틱: {}개), 페이지: {}/{}, 카테고리 분포: {}", 
                    totalCount, keywordCount, totalCount - keywordCount, page + 1, 
                    (int) Math.ceil((double) totalCount / size), categoryCounts);
            
            // 상세 로그는 debug 레벨로
            log.debug("최종 검색 결과 순서 (상위 10개): {}", 
                    items.stream()
                            .limit(10)
                            .map(item -> String.format("%d:%s", item.id(), item.itemName()))
                            .collect(Collectors.joining(", ")));
        }

        return LostItemListDto.builder()
                .items(items)
                .totalCount(totalCount) // 실제 전체 개수 반환
                .page(page)
                .size(size)
                .build();
    }
    
    /**
     * 키워드 기반 검색 (제목/설명/브랜드에 검색어 포함)
     * 필터는 적용하지 않음 (나중에 통합 필터링에서 처리)
     */
    private List<LostItem> searchByKeyword(String keyword, SearchLostItemRequest request, Double locationRadius) {
        if (keyword == null || keyword.trim().isEmpty()) {
            return List.of();
        }
        
        String searchKeyword = keyword.toLowerCase().trim();
        
        // 특수문자 이스케이프 처리
        char escapeChar = '!';
        String escapedKeyword = searchKeyword
                .replace("!", "!!")
                .replace("%", "!%")
                .replace("_", "!_");
        String pattern = "%" + escapedKeyword + "%";
        
        // 키워드 매칭만 수행 (필터는 나중에 통합 필터링에서 처리)
        Specification<LostItem> spec = (root, query, cb) -> {
            List<Predicate> predicates = new java.util.ArrayList<>();
            
            // 키워드 매칭: itemName, description, brand 중 하나라도 포함
            predicates.add(cb.or(
                    cb.like(cb.lower(root.get("itemName")), pattern, escapeChar),
                    cb.like(cb.lower(root.get("description")), pattern, escapeChar),
                    cb.and(
                            cb.isNotNull(root.get("brand")),
                            cb.like(cb.lower(root.get("brand")), pattern, escapeChar)
                    )
            ));
            
            return cb.and(predicates.toArray(new Predicate[0]));
        };
        
        // DB 레벨에서 정렬 및 제한 (메모리 정렬 최소화)
        Pageable pageable = PageRequest.of(0, 100, Sort.by(Sort.Direction.DESC, "foundDate"));
        Page<LostItem> itemPage = lostItemRepository.findAll(spec, pageable);
        List<LostItem> items = itemPage.getContent();
        
        return items;
    }

    /**
     * 필터가 있는지 확인
     */
    private boolean hasFilters(SearchLostItemRequest request) {
        boolean hasLocation = (request.location() != null && !request.location().trim().isEmpty())
                || (request.locations() != null && !request.locations().isEmpty());
        // locationRadius만 있고 장소명이 없으면 필터로 간주하지 않음 (필터링 불가능)
        return request.category() != null
                || hasLocation
                || (request.brand() != null && !request.brand().trim().isEmpty())
                || request.foundDateAfter() != null;
    }

    /**
     * 아이템이 필터 조건에 맞는지 확인
     * 장소 필터링은 좌표 기반 반경 필터링을 사용
     * 여러 장소가 있는 경우 OR 조건 (하나라도 만족하면 통과)
     * @param item 분실물 아이템
     * @param request 검색 요청
     * @param locationPlaceResults 미리 변환된 장소 좌표 리스트
     * @param locationRadius 반경 (미터)
     */
    private boolean matchesFilters(LostItem item, SearchLostItemRequest request, 
                                   List<TmapApiService.TmapPlaceResult> locationPlaceResults,
                                   Double locationRadius) {
        // 카테고리 필터
        if (request.category() != null && !item.getCategory().equals(request.category())) {
            return false;
        }

        // 장소 필터 (좌표 기반 반경 필터링)
        // 여러 장소가 있는 경우 OR 조건 (하나라도 만족하면 통과)
        boolean locationMatches = false;
        
        // 장소 필터가 있는지 확인
        boolean hasLocationFilter = (request.locations() != null && !request.locations().isEmpty()) ||
                                    (request.location() != null && !request.location().trim().isEmpty());
        
        // locationRadius만 있고 장소명이 없으면 필터링 불가능하므로 통과
        if (!hasLocationFilter) {
            locationMatches = true;
        }
        // 여러 장소가 있는 경우
        else if (locationPlaceResults != null && !locationPlaceResults.isEmpty()) {
            double radius = locationRadius != null ? locationRadius : 10000.0; // 기본값 10km
            
            // 아이템 좌표가 없으면 필터링 실패 (거리 필터가 설정된 경우)
            if (item.getLatitude() == null || item.getLongitude() == null) {
                // 거리 필터가 설정되어 있으면 좌표가 없으면 실패
                if (locationRadius != null) {
                    log.debug("장소 필터 실패: itemId={}, itemLocation={}, 좌표 없음 (거리 필터 설정됨)", 
                            item.getId(), item.getLocation());
                    return false;
                }
                // 거리 필터가 없으면 문자열 일치로 폴백
                List<String> locationsToCheck = request.locations() != null && !request.locations().isEmpty() 
                    ? request.locations() 
                    : (request.location() != null && !request.location().trim().isEmpty() 
                        ? List.of(request.location()) 
                        : List.of());
                
                for (String loc : locationsToCheck) {
                    if (loc != null && !loc.trim().isEmpty() && item.getLocation() != null) {
                        String locationKeyword = loc.toLowerCase().trim();
                        String itemLocation = item.getLocation().toLowerCase();
                        
                        // 키워드가 '역', '구', '동' 등으로 끝나는 경우 이를 제거하고 더 넓게 검색
                        String locationKeywordWithoutSuffix = locationKeyword;
                        if (locationKeyword.endsWith("역")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("구")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("동")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        }
                        
                        boolean matches = itemLocation.contains(locationKeyword);
                        if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                            matches = itemLocation.contains(locationKeywordWithoutSuffix);
                        }
                        if (matches) {
                            log.debug("장소 필터 통과 (문자열 일치): itemId={}, itemLocation={}, 검색어={}", 
                                    item.getId(), item.getLocation(), loc);
                            return true; // 문자열 일치로 통과
                        }
                    }
                }
                // 문자열 일치도 실패하면 필터링 실패
                log.debug("장소 필터 실패: itemId={}, itemLocation={}, 좌표 없음, 문자열 일치 실패", 
                        item.getId(), item.getLocation());
                return false;
            }
            
            // 여러 장소 중 하나라도 반경 내에 있으면 통과
            for (TmapApiService.TmapPlaceResult placeResult : locationPlaceResults) {
                if (placeResult != null) {
                    double distance = calculateHaversineDistance(
                            placeResult.getLatitude(), placeResult.getLongitude(),
                            item.getLatitude(), item.getLongitude()
                    );
                    log.debug("거리 계산: 장소=({}, {}), 아이템=({}, {}), 거리={}m, 반경={}m, 통과={}", 
                            placeResult.getLatitude(), placeResult.getLongitude(),
                            item.getLatitude(), item.getLongitude(),
                            distance, radius, distance <= radius);
                    if (distance <= radius) {
                        locationMatches = true;
                        log.debug("장소 필터 통과: itemId={}, itemLocation={}, 장소 좌표=({}, {}), 거리={}m", 
                                item.getId(), item.getLocation(), 
                                placeResult.getLatitude(), placeResult.getLongitude(), distance);
                        break; // 하나라도 만족하면 통과
                    }
                }
            }
            
            if (!locationMatches) {
                log.debug("장소 필터 실패: itemId={}, itemLocation={}, 좌표=({}, {}), 모든 장소에서 반경 밖", 
                        item.getId(), item.getLocation(), item.getLatitude(), item.getLongitude());
            }
            
            // 좌표 기반 필터링이 실패했을 때, 거리 필터가 설정되지 않은 경우에만 문자열 일치로 폴백
            if (!locationMatches && locationRadius == null) {
                // 거리 필터가 없으면 문자열 일치로 폴백
                List<String> locationsToCheck = request.locations() != null && !request.locations().isEmpty() 
                    ? request.locations() 
                    : (request.location() != null && !request.location().trim().isEmpty() 
                        ? List.of(request.location()) 
                        : List.of());
                
                for (String loc : locationsToCheck) {
                    if (loc != null && !loc.trim().isEmpty() && item.getLocation() != null) {
                        String locationKeyword = loc.toLowerCase().trim();
                        String itemLocation = item.getLocation().toLowerCase();
                        
                        // 키워드가 '역', '구', '동' 등으로 끝나는 경우 이를 제거하고 더 넓게 검색
                        String locationKeywordWithoutSuffix = locationKeyword;
                        if (locationKeyword.endsWith("역")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("구")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("동")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        }
                        
                        boolean matches = itemLocation.contains(locationKeyword);
                        if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                            matches = itemLocation.contains(locationKeywordWithoutSuffix);
                        }
                        if (matches) {
                            locationMatches = true;
                            log.debug("장소 필터 통과 (문자열 일치 폴백): itemId={}, itemLocation={}, 검색어={}", 
                                    item.getId(), item.getLocation(), loc);
                            break;
                        }
                    }
                }
            }
        }
        // 좌표 변환이 실패한 경우 (locationPlaceResults가 비어있음)
        else if (hasLocationFilter) {
            // 거리 필터가 설정되어 있으면 좌표 변환 실패 시 필터링 실패
            if (locationRadius != null) {
                log.warn("장소 필터 실패: 좌표 변환 실패 (거리 필터 설정됨), itemId={}, itemLocation={}", 
                        item.getId(), item.getLocation());
                return false;
            }
            // 거리 필터가 없으면 문자열 일치로 폴백
            List<String> locationsToCheck = request.locations() != null && !request.locations().isEmpty() 
                ? request.locations() 
                : (request.location() != null && !request.location().trim().isEmpty() 
                    ? List.of(request.location()) 
                    : List.of());
            
            for (String loc : locationsToCheck) {
                if (loc != null && !loc.trim().isEmpty() && item.getLocation() != null) {
                    String locationKeyword = loc.toLowerCase().trim();
                    String itemLocation = item.getLocation().toLowerCase();
                    
                    // 키워드가 '역', '구', '동' 등으로 끝나는 경우 이를 제거하고 더 넓게 검색
                    String locationKeywordWithoutSuffix = locationKeyword;
                    if (locationKeyword.endsWith("역")) {
                        locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                    } else if (locationKeyword.endsWith("구")) {
                        locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                    } else if (locationKeyword.endsWith("동")) {
                        locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                    }
                    
                    boolean matches = itemLocation.contains(locationKeyword);
                    if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                        matches = itemLocation.contains(locationKeywordWithoutSuffix);
                    }
                    if (matches) {
                        locationMatches = true;
                        log.debug("장소 필터 통과 (문자열 일치, 좌표 변환 실패): itemId={}, itemLocation={}, 검색어={}", 
                                item.getId(), item.getLocation(), loc);
                        break;
                    }
                }
            }
            
            if (!locationMatches) {
                log.debug("장소 필터 실패: 좌표 변환 실패, 문자열 일치도 실패, itemId={}, itemLocation={}", 
                        item.getId(), item.getLocation());
            }
        }
        // 단일 장소가 있는 경우 (하위 호환성)
        else if (request.location() != null && !request.location().trim().isEmpty()) {
            // 이미 변환된 좌표가 있으면 사용, 없으면 변환 시도 (폴백)
            TmapApiService.TmapPlaceResult placeResult = null;
            if (locationPlaceResults != null && !locationPlaceResults.isEmpty()) {
                placeResult = locationPlaceResults.get(0);
            }
            if (placeResult == null) {
                // 폴백: 변환되지 않은 경우에만 호출 (일반적으로는 발생하지 않아야 함)
                log.warn("matchesFilters에서 장소 좌표가 전달되지 않아 변환 시도: {}", request.location());
                placeResult = tmapApiService.searchPlace(request.location().trim());
            }
            
            if (placeResult != null) {
                // 좌표 기반 반경 필터링
                if (item.getLatitude() == null || item.getLongitude() == null) {
                    // 좌표가 없으면 거리 필터가 설정된 경우 실패, 없으면 문자열 일치로 폴백
                    if (locationRadius != null) {
                        return false; // 거리 필터가 설정된 경우 좌표가 없으면 실패
                    }
                    // 거리 필터가 없으면 문자열 일치로 폴백
                    String locationKeyword = request.location().toLowerCase().trim();
                    if (item.getLocation() != null) {
                        String itemLocation = item.getLocation().toLowerCase();
                        String locationKeywordWithoutSuffix = locationKeyword;
                        if (locationKeyword.endsWith("역")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("구")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        } else if (locationKeyword.endsWith("동")) {
                            locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                        }
                        boolean matches = itemLocation.contains(locationKeyword);
                        if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                            matches = itemLocation.contains(locationKeywordWithoutSuffix);
                        }
                        locationMatches = matches;
                    } else {
                        return false;
                    }
                } else {
                    double distance = calculateHaversineDistance(
                            placeResult.getLatitude(), placeResult.getLongitude(),
                            item.getLatitude(), item.getLongitude()
                    );
                    double radius = locationRadius != null ? locationRadius : 10000.0; // 기본값 10km
                    if (distance <= radius) {
                        locationMatches = true;
                    } else if (locationRadius == null) {
                        // 거리 필터가 없으면 문자열 일치로 폴백
                        String locationKeyword = request.location().toLowerCase().trim();
                        if (item.getLocation() != null) {
                            String itemLocation = item.getLocation().toLowerCase();
                            String locationKeywordWithoutSuffix = locationKeyword;
                            if (locationKeyword.endsWith("역")) {
                                locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                            } else if (locationKeyword.endsWith("구")) {
                                locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                            } else if (locationKeyword.endsWith("동")) {
                                locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                            }
                            boolean matches = itemLocation.contains(locationKeyword);
                            if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                                matches = itemLocation.contains(locationKeywordWithoutSuffix);
                            }
                            locationMatches = matches;
                        }
                    }
                }
            } else {
                // 좌표 변환 실패 시: 거리 필터가 설정된 경우 실패, 없으면 문자열 일치로 폴백
                if (locationRadius != null) {
                    return false; // 거리 필터가 설정된 경우 좌표 변환 실패하면 실패
                }
                // 거리 필터가 없으면 문자열 일치 방식으로 폴백
                String locationKeyword = request.location().toLowerCase().trim();
                
                // 키워드가 '역', '구', '동' 등으로 끝나는 경우 이를 제거하고 더 넓게 검색
                String locationKeywordWithoutSuffix = locationKeyword;
                if (locationKeyword.endsWith("역")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                } else if (locationKeyword.endsWith("구")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                } else if (locationKeyword.endsWith("동")) {
                    locationKeywordWithoutSuffix = locationKeyword.substring(0, locationKeyword.length() - 1);
                }
                
                // 원본 키워드 또는 접미사 제거한 키워드가 포함되어 있으면 통과
                if (item.getLocation() != null) {
                    String itemLocation = item.getLocation().toLowerCase();
                    boolean matches = itemLocation.contains(locationKeyword);
                    if (!matches && !locationKeywordWithoutSuffix.equals(locationKeyword) && locationKeywordWithoutSuffix.length() > 0) {
                        matches = itemLocation.contains(locationKeywordWithoutSuffix);
                    }
                    locationMatches = matches;
                } else {
                    return false;
                }
            }
        } else {
            // 장소 필터가 없으면 통과
            locationMatches = true;
        }
        
        // 장소 필터가 있는데 매칭되지 않으면 실패
        if ((request.location() != null && !request.location().trim().isEmpty()) || 
            (request.locations() != null && !request.locations().isEmpty())) {
            if (!locationMatches) {
                return false;
            }
        }

        // 브랜드 필터 (부분 일치)
        // itemName, description, brand 필드 모두에서 검색
        if (request.brand() != null && !request.brand().trim().isEmpty()) {
            String brand = request.brand().toLowerCase().trim();
            
            // itemName에서 검색
            boolean matchesItemName = item.getItemName() != null && 
                    item.getItemName().toLowerCase().contains(brand);
            
            // description에서 검색
            boolean matchesDescription = item.getDescription() != null && 
                    item.getDescription().toLowerCase().contains(brand);
            
            // brand 필드에서 검색
            boolean matchesBrand = item.getBrand() != null && 
                    item.getBrand().toLowerCase().contains(brand);
            
            // 하나라도 매치되면 통과
            if (!matchesItemName && !matchesDescription && !matchesBrand) {
                log.debug("브랜드 필터 실패: itemId={}, itemName='{}', description='{}', brand='{}', 검색 브랜드='{}'", 
                        item.getId(), item.getItemName(), 
                        item.getDescription() != null ? item.getDescription().substring(0, Math.min(50, item.getDescription().length())) : null,
                        item.getBrand(), brand);
                return false;
            }
            
            log.debug("브랜드 필터 통과: itemId={}, matchesItemName={}, matchesDescription={}, matchesBrand={}, 검색 브랜드='{}'", 
                    item.getId(), matchesItemName, matchesDescription, matchesBrand, brand);
        }

        // 날짜 필터 (해당 날짜 이후)
        if (request.foundDateAfter() != null) {
            if (item.getFoundDate() == null || item.getFoundDate().isBefore(request.foundDateAfter())) {
                return false;
            }
        }

        // 모든 필터를 통과했으면 true 반환
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
