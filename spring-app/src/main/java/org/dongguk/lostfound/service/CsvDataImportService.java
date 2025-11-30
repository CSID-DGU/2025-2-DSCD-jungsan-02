package org.dongguk.lostfound.service;

import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.domain.custody.CustodyLocation;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.repository.CustodyLocationRepository;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

/**
 * CSV 파일에서 데이터를 임포트하는 배치 서비스
 */
@Slf4j
@Service
public class CsvDataImportService {
    private final CustodyLocationRepository custodyLocationRepository;
    private final UserRepository userRepository;
    private final LostItemRepository lostItemRepository;
    private final PasswordEncoder passwordEncoder;
    private final FlaskApiService flaskApiService;
    private final ThreadPoolTaskExecutor embeddingExecutor;
    
    public CsvDataImportService(
            CustodyLocationRepository custodyLocationRepository,
            UserRepository userRepository,
            LostItemRepository lostItemRepository,
            PasswordEncoder passwordEncoder,
            FlaskApiService flaskApiService,
            @Qualifier("embeddingExecutor") ThreadPoolTaskExecutor embeddingExecutor) {
        this.custodyLocationRepository = custodyLocationRepository;
        this.userRepository = userRepository;
        this.lostItemRepository = lostItemRepository;
        this.passwordEncoder = passwordEncoder;
        this.flaskApiService = flaskApiService;
        this.embeddingExecutor = embeddingExecutor;
    }
    
    private static final String DEFAULT_PASSWORD = "1234";
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");
    
    // 저장할 카테고리만 필터링
    private static final Set<String> ALLOWED_CATEGORIES = Set.of("지갑", "가방", "의류");
    
    // 카테고리 매핑
    private static final Map<String, ItemCategory> CATEGORY_MAP = Map.of(
        "지갑", ItemCategory.WALLET,
        "가방", ItemCategory.BAG,
        "의류", ItemCategory.CLOTHING
    );

    /**
     * 보관소 위치 CSV 파일 임포트
     * 형식: 보관위치,위도,경도
     */
    @Transactional
    public ImportResult importCustodyLocations(String csvFilePath) {
        log.info("보관소 위치 데이터 임포트 시작: {}", csvFilePath);
        
        int savedCount = 0;
        int skippedCount = 0;
        int errorCount = 0;
        
        try (BufferedReader reader = new BufferedReader(
                new FileReader(csvFilePath, StandardCharsets.UTF_8))) {
            
            String line;
            boolean isFirstLine = true;
            List<CustodyLocation> locations = new ArrayList<>();
            
            while ((line = reader.readLine()) != null) {
                // 헤더 스킵
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                
                try {
                    String[] parts = line.split(",");
                    if (parts.length < 3) {
                        skippedCount++;
                        continue;
                    }
                    
                    String name = parts[0].trim();
                    Double latitude = Double.parseDouble(parts[1].trim());
                    Double longitude = Double.parseDouble(parts[2].trim());
                    
                    // 이미 존재하는지 확인
                    if (custodyLocationRepository.findByName(name).isEmpty()) {
                        CustodyLocation location = CustodyLocation.create(name, latitude, longitude, 0);
                        locations.add(location);
                        savedCount++;
                    } else {
                        skippedCount++;
                    }
                } catch (Exception e) {
                    log.warn("CSV 파싱 오류: {}", line, e);
                    errorCount++;
                }
            }
            
            // 배치 저장
            if (!locations.isEmpty()) {
                custodyLocationRepository.saveAll(locations);
                log.info("보관소 위치 데이터 임포트 완료: {}개 저장, {}개 스킵, {}개 오류", 
                    savedCount, skippedCount, errorCount);
            }
            
        } catch (IOException e) {
            log.error("보관소 위치 데이터 임포트 실패: {}", csvFilePath, e);
            throw new RuntimeException("보관소 위치 데이터 임포트 실패", e);
        }
        
        return new ImportResult(savedCount, skippedCount, errorCount);
    }

    /**
     * 분실물 데이터 CSV 파일 임포트
     * 형식: MANAGE_ID,COLOR_NM,CUSTODY_PLC,ACQUSTN_THNG_PHOTO_URL,THNG_DTLS,NOTI_CONT,ACQUSTN_DE,SN_ID,THNG_CLASS_LARGE,THNG_CLASS_SMALL,LOCAL_IMAGE_PATH
     * 
     * 개선사항:
     * - 배치마다 트랜잭션을 분리하여 DB에 즉시 커밋
     * - 임베딩 생성을 비동기 병렬로 처리하여 성능 향상
     */
    public ImportResult importLostItems(String csvFilePath) {
        log.info("분실물 데이터 임포트 시작: {}", csvFilePath);
        
        int savedCount = 0;
        int skippedCount = 0;
        int errorCount = 0;
        
        // 보관소별 User 맵 (이미 생성된 User 재사용)
        Map<String, User> custodyUserMap = new HashMap<>();
        
        // 비동기 임베딩 생성 작업들을 추적하기 위한 리스트
        List<CompletableFuture<Void>> embeddingFutures = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(
                new FileReader(csvFilePath, StandardCharsets.UTF_8))) {
            
            String line;
            boolean isFirstLine = true;
            List<LostItem> lostItems = new ArrayList<>();
            int batchSize = 200; // 배치 크기 (500에서 200으로 감소 - 메모리 및 트랜잭션 부담 감소)
            long processedLines = 0;
            
            while ((line = reader.readLine()) != null) {
                // 헤더 스킵
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                
                processedLines++;
                
                try {
                    // CSV 파싱 (쉼표로 분리, 하지만 값 안에 쉼표가 있을 수 있으므로 주의)
                    String[] parts = parseCsvLine(line);
                    if (parts.length < 11) {
                        skippedCount++;
                        continue;
                    }
                    
                    String custodyPlace = parts[2].trim(); // CUSTODY_PLC
                    String imageUrl = parts[3].trim(); // ACQUSTN_THNG_PHOTO_URL
                    String itemName = parts[4].trim(); // THNG_DTLS
                    String description = parts[5].trim(); // NOTI_CONT
                    String foundDateStr = parts[6].trim(); // ACQUSTN_DE
                    String categoryStr = parts[8].trim(); // THNG_CLASS_LARGE
                    
                    // 필수 필드 검증
                    if (custodyPlace.isEmpty() || itemName.isEmpty() || description.isEmpty() || 
                        foundDateStr.isEmpty() || categoryStr.isEmpty()) {
                        skippedCount++;
                        continue;
                    }
                    
                    // 이미지 URL이 있으면 빠르게 검증 (HEAD 요청으로 존재 여부만 확인)
                    // 이미지 다운로드 실패 시 DB 저장을 방지
                    if (!imageUrl.isEmpty()) {
                        if (!isImageUrlValid(imageUrl)) {
                            if (processedLines % 100 == 0) { // 100개마다 한 번씩만 로그 출력
                                log.debug("이미지 URL 접근 불가, 스킵: {} (처리된 라인: {})", imageUrl, processedLines);
                            }
                            skippedCount++;
                            continue; // 이미지가 없으면 DB에 저장하지 않음
                        }
                    }
                    
                    // 카테고리 필터링 (지갑, 가방, 의류만)
                    if (!ALLOWED_CATEGORIES.contains(categoryStr)) {
                        skippedCount++;
                        continue;
                    }
                    
                    // 카테고리 매핑
                    ItemCategory category = CATEGORY_MAP.get(categoryStr);
                    if (category == null) {
                        skippedCount++;
                        continue;
                    }
                    
                    // 날짜 파싱
                    LocalDate foundDate;
                    try {
                        foundDate = LocalDate.parse(foundDateStr, DATE_FORMATTER);
                    } catch (DateTimeParseException e) {
                        log.warn("날짜 파싱 오류: {}", foundDateStr);
                        skippedCount++;
                        continue;
                    }
                    
                    // User 가져오기 또는 생성
                    User user = custodyUserMap.computeIfAbsent(custodyPlace, place -> {
                        return userRepository.findByLoginId(place)
                            .orElseGet(() -> {
                                // User가 없으면 생성
                                String encodedPassword = passwordEncoder.encode(DEFAULT_PASSWORD);
                                User newUser = User.create(place, encodedPassword);
                                return userRepository.save(newUser);
                            });
                    });
                    
                    // 보관소 위치 정보 가져오기 (위도, 경도)
                    CustodyLocation custodyLocation = custodyLocationRepository.findByName(custodyPlace)
                        .orElse(null);
                    
                    Double latitude = custodyLocation != null ? custodyLocation.getLatitude() : null;
                    Double longitude = custodyLocation != null ? custodyLocation.getLongitude() : null;
                    
                    // LostItem 생성
                    LostItem lostItem = LostItem.create(
                        itemName,
                        category,
                        description,
                        foundDate,
                        custodyPlace, // location
                        latitude,
                        longitude,
                        null, // brand
                        imageUrl.isEmpty() ? null : imageUrl,
                        null, // embeddingId (나중에 생성)
                        user
                    );
                    
                    lostItems.add(lostItem);
                    savedCount++;
                    
                    // 배치 저장 (각 배치마다 트랜잭션 분리)
                    if (lostItems.size() >= batchSize) {
                        List<LostItem> savedItems = saveBatchWithTransaction(lostItems);
                        
                        // 저장된 아이템들에 대해 임베딩 생성 (비동기 병렬 처리)
                        List<CompletableFuture<Void>> batchFutures = createEmbeddingsAsync(savedItems);
                        embeddingFutures.addAll(batchFutures);
                        
                        lostItems.clear();
                        log.info("분실물 데이터 배치 저장 완료: {}개 저장됨 (처리된 라인: {}개, 총 저장: {}개)", 
                            savedItems.size(), processedLines, savedCount);
                    }
                    
                } catch (Exception e) {
                    log.warn("CSV 파싱 오류: {}", line, e);
                    errorCount++;
                }
            }
            
            // 남은 데이터 저장
            if (!lostItems.isEmpty()) {
                List<LostItem> savedItems = saveBatchWithTransaction(lostItems);
                
                // 저장된 아이템들에 대해 임베딩 생성 (비동기 병렬 처리)
                List<CompletableFuture<Void>> batchFutures = createEmbeddingsAsync(savedItems);
                embeddingFutures.addAll(batchFutures);
                
                log.info("분실물 데이터 최종 배치 저장 완료: {}개 저장됨", savedItems.size());
            }
            
            // 모든 임베딩 생성 작업이 완료될 때까지 대기 (선택사항)
            // DB 저장은 이미 완료되었으므로, 임베딩 생성은 백그라운드에서 계속 진행 가능
            log.info("DB 저장 완료. 임베딩 생성 작업 {}개가 백그라운드에서 진행 중...", embeddingFutures.size());
            
            // 모든 임베딩 작업 완료 대기 (타임아웃 방지를 위해 선택적으로 사용)
            CompletableFuture.allOf(embeddingFutures.toArray(new CompletableFuture[0]))
                .whenComplete((result, throwable) -> {
                    if (throwable != null) {
                        log.error("일부 임베딩 생성 작업에서 오류 발생", throwable);
                    } else {
                        log.info("모든 임베딩 생성 작업 완료");
                    }
                });
            
            log.info("분실물 데이터 임포트 완료: {}개 저장, {}개 스킵, {}개 오류 (총 {}줄 처리)", 
                savedCount, skippedCount, errorCount, processedLines);
            
        } catch (IOException e) {
            log.error("분실물 데이터 임포트 실패: {}", csvFilePath, e);
            throw new RuntimeException("분실물 데이터 임포트 실패", e);
        }
        
        return new ImportResult(savedCount, skippedCount, errorCount);
    }
    
    /**
     * 배치 단위로 트랜잭션을 분리하여 저장
     * 각 배치마다 즉시 커밋되므로 DB에 바로 반영됨
     */
    @Transactional
    public List<LostItem> saveBatchWithTransaction(List<LostItem> lostItems) {
        return lostItemRepository.saveAll(lostItems);
    }
    
    /**
     * 임베딩 생성을 비동기 병렬로 처리
     * 배치 API를 사용하여 성능 향상 (네트워크 오버헤드 감소)
     * 각 배치마다 여러 아이템을 한 번에 Flask 서버로 전송
     */
    private List<CompletableFuture<Void>> createEmbeddingsAsync(List<LostItem> savedItems) {
        // 배치 크기 설정 (한 번에 처리할 아이템 수)
        int batchSize = 20; // 배치 크기 조정 가능
        
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        // 배치 단위로 나누어 처리
        for (int i = 0; i < savedItems.size(); i += batchSize) {
            int end = Math.min(i + batchSize, savedItems.size());
            List<LostItem> batch = savedItems.subList(i, end);
            
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    // 배치 API 사용
                    flaskApiService.createEmbeddingsBatch(batch);
                    log.debug("배치 임베딩 생성 완료: {}개 아이템", batch.size());
                } catch (Exception e) {
                    log.warn("배치 임베딩 생성 실패, 개별 처리로 전환: {}", e.getMessage());
                    // 배치 실패 시 개별 처리로 fallback
                    for (LostItem item : batch) {
                        try {
                            flaskApiService.createEmbeddingFromUrl(
                                item.getId(),
                                item.getDescription(),
                                item.getImageUrl()
                            );
                        } catch (Exception ex) {
                            log.warn("임베딩 생성 실패 (나중에 재시도 가능): itemId={}", item.getId(), ex);
                        }
                    }
                }
            }, embeddingExecutor);
            
            futures.add(future);
        }
        
        return futures;
    }
    
    /**
     * 이미지 URL이 유효한지 빠르게 검증 (HEAD 요청 사용)
     * 이미지 다운로드 실패 시 DB 저장을 방지하기 위함
     */
    private boolean isImageUrlValid(String imageUrl) {
        if (imageUrl == null || imageUrl.isEmpty()) {
            return false;
        }
        
        try {
            URI uri = URI.create(imageUrl);
            HttpURLConnection connection = (HttpURLConnection) uri.toURL().openConnection();
            connection.setRequestMethod("HEAD"); // HEAD 요청으로 빠르게 검증
            connection.setConnectTimeout(5000); // 5초 연결 타임아웃
            connection.setReadTimeout(5000); // 5초 읽기 타임아웃
            connection.setRequestProperty("User-Agent", "Mozilla/5.0");
            
            int responseCode = connection.getResponseCode();
            String contentType = connection.getContentType();
            
            connection.disconnect();
            
            // 200 OK이고 이미지 타입인지 확인
            boolean isValid = responseCode == 200 && 
                             contentType != null && 
                             contentType.startsWith("image/");
            
            if (!isValid) {
                log.debug("이미지 URL 검증 실패: {} (응답 코드: {}, 타입: {})", 
                    imageUrl, responseCode, contentType);
            }
            
            return isValid;
            
        } catch (Exception e) {
            log.debug("이미지 URL 검증 중 오류: {} - {}", imageUrl, e.getMessage());
            return false;
        }
    }
    
    /**
     * CSV 라인 파싱 (값 안에 쉼표가 있을 수 있음)
     * 간단한 구현: 쉼표로 분리하되, 따옴표로 감싸진 값은 처리하지 않음
     */
    private String[] parseCsvLine(String line) {
        List<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (char c : line.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                result.add(current.toString());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        result.add(current.toString());
        
        return result.toArray(new String[0]);
    }
    
    /**
     * 임포트 결과를 담는 클래스
     */
    public static class ImportResult {
        private final int savedCount;
        private final int skippedCount;
        private final int errorCount;
        
        public ImportResult(int savedCount, int skippedCount, int errorCount) {
            this.savedCount = savedCount;
            this.skippedCount = skippedCount;
            this.errorCount = errorCount;
        }
        
        public int getSavedCount() {
            return savedCount;
        }
        
        public int getSkippedCount() {
            return skippedCount;
        }
        
        public int getErrorCount() {
            return errorCount;
        }
        
        @Override
        public String toString() {
            return String.format("저장: %d개, 스킵: %d개, 오류: %d개", 
                savedCount, skippedCount, errorCount);
        }
    }
}

