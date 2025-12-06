package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class FlaskApiService {
    private final RestClient flaskRestClient;

    /**
     * Flask AI 서버에 임베딩 생성 요청
     */
    public void createEmbedding(Long itemId, String itemName, String description, MultipartFile image) {
        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("item_id", itemId.toString());
            builder.part("item_name", itemName != null ? itemName : "");  // 분실물 제목 추가
            builder.part("description", description != null ? description : "");
            
            if (image != null && !image.isEmpty()) {
                builder.part("image", new ByteArrayResource(image.getBytes()) {
                    @Override
                    public String getFilename() {
                        return image.getOriginalFilename();
                    }
                }, MediaType.IMAGE_JPEG);
            }

            Map<String, Object> response = flaskRestClient.post()
                    .uri("/api/v1/embedding/create")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(builder.build())
                    .retrieve()
                    .body(Map.class);

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Successfully created embedding for item {}", itemId);
                return;
            }
            
            throw new RuntimeException("Failed to create embedding: " + response);
            
        } catch (IOException e) {
            log.error("Error reading image file for item {}", itemId, e);
            throw new RuntimeException("Failed to create embedding", e);
        } catch (Exception e) {
            log.warn("Flask AI 서버에 연결할 수 없습니다. 임베딩 생성을 건너뜁니다. itemId: {}", itemId);
            // Flask 서버가 꺼져있어도 계속 진행
        }
    }

    /**
     * Flask AI 서버에 검색 요청 (item_id 리스트와 점수 반환)
     */
    public static class SearchResult {
        private final List<Long> itemIds;
        private final List<Double> scores;
        
        public SearchResult(List<Long> itemIds, List<Double> scores) {
            this.itemIds = itemIds;
            this.scores = scores;
        }
        
        public List<Long> getItemIds() {
            return itemIds;
        }
        
        public List<Double> getScores() {
            return scores;
        }
    }
    
    /**
     * Flask AI 서버에 검색 요청 (item_id 리스트 반환) - 하위 호환성 유지
     */
    public List<Long> searchSimilarItems(String query, Integer topK) {
        SearchResult result = searchSimilarItemsWithScores(query, topK);
        return result.getItemIds();
    }
    
    /**
     * Flask AI 서버에 검색 요청 (item_id 리스트와 점수 반환)
     */
    public SearchResult searchSimilarItemsWithScores(String query, Integer topK) {
        try {
            Map<String, Object> request = Map.of(
                    "query", query,
                    "top_k", topK != null ? topK : 10
            );

            Map<String, Object> response = flaskRestClient.post()
                    .uri("/api/v1/embedding/search")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(Map.class);

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                List<Integer> itemIds = (List<Integer>) response.get("item_ids");
                List<Double> scores = (List<Double>) response.get("scores");
                
                List<Long> longItemIds = itemIds != null ? 
                        itemIds.stream().map(Long::valueOf).toList() : 
                        new ArrayList<>();
                List<Double> doubleScores = scores != null ? 
                        scores : 
                        new ArrayList<>();
                
                // 디버깅: 유사도 점수 확인
                if (!doubleScores.isEmpty()) {
                    log.info("Flask 검색 결과 - 상위 10개 유사도 점수: {}", 
                            doubleScores.stream()
                                    .limit(10)
                                    .map(score -> String.format("%.4f", score))
                                    .collect(Collectors.joining(", ")));
                    double maxScore = doubleScores.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
                    double minScore = doubleScores.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
                    log.info("Flask 검색 결과 - 점수 범위: 최고={}, 최저={}", 
                            String.format("%.4f", maxScore),
                            String.format("%.4f", minScore));
                }
                
                return new SearchResult(longItemIds, doubleScores);
            }
            
            throw new RuntimeException("Failed to search embeddings: " + response);
            
        } catch (Exception e) {
            log.warn("Flask AI 서버에 연결할 수 없습니다. 빈 결과를 반환합니다. query: {}", query);
            // Flask 서버가 꺼져있을 때 빈 리스트 반환 (fallback)
            return new SearchResult(new ArrayList<>(), new ArrayList<>());
        }
    }

    /**
     * Flask AI 서버에 임베딩 생성 요청 (이미지 URL 사용)
     * CSV 임포트된 데이터의 이미지 URL에서 이미지를 다운로드하여 임베딩 생성
     */
    public void createEmbeddingFromUrl(Long itemId, String itemName, String description, String imageUrl) {
        if (imageUrl == null || imageUrl.isEmpty()) {
            // 이미지가 없으면 텍스트만으로 임베딩 생성
            createEmbedding(itemId, itemName, description, null);
            return;
        }
        
        try {
            // 이미지 URL에서 이미지 다운로드
            URI uri = URI.create(imageUrl);
            byte[] imageBytes;
            
            // HTTP 연결 설정 (타임아웃 및 User-Agent 설정)
            HttpURLConnection connection = (HttpURLConnection) uri.toURL().openConnection();
            connection.setConnectTimeout(10000); // 10초 연결 타임아웃
            connection.setReadTimeout(30000); // 30초 읽기 타임아웃
            connection.setRequestProperty("User-Agent", "Mozilla/5.0"); // 일부 서버에서 User-Agent 필요
            
            try (InputStream in = connection.getInputStream()) {
                imageBytes = in.readAllBytes();
            }
            
            // 이미지 크기 제한 (20MB)
            if (imageBytes.length > 20 * 1024 * 1024) {
                log.warn("이미지 파일이 너무 큽니다 ({}MB). 스킵합니다: {}", 
                    imageBytes.length / (1024 * 1024), imageUrl);
                // 텍스트만으로 임베딩 생성
                createEmbedding(itemId, itemName, description, null);
                return;
            }
            
            // 임시 MultipartFile 생성
            MultipartFile imageFile = new MultipartFile() {
                @Override
                public String getName() {
                    return "image";
                }
                
                @Override
                public String getOriginalFilename() {
                    String filename = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);
                    // 확장자가 없으면 기본값 추가
                    if (!filename.contains(".")) {
                        filename += ".jpg";
                    }
                    return filename;
                }
                
                @Override
                public String getContentType() {
                    String filename = getOriginalFilename().toLowerCase();
                    if (filename.endsWith(".png")) {
                        return "image/png";
                    } else if (filename.endsWith(".gif")) {
                        return "image/gif";
                    }
                    return "image/jpeg";
                }
                
                @Override
                public boolean isEmpty() {
                    return imageBytes.length == 0;
                }
                
                @Override
                public long getSize() {
                    return imageBytes.length;
                }
                
                @Override
                public byte[] getBytes() throws IOException {
                    return imageBytes;
                }
                
                @Override
                public InputStream getInputStream() throws IOException {
                    return new java.io.ByteArrayInputStream(imageBytes);
                }
                
                @Override
                public void transferTo(java.io.File dest) throws IOException, IllegalStateException {
                    java.nio.file.Files.write(dest.toPath(), imageBytes);
                }
            };
            
            // 기존 메서드 호출
            createEmbedding(itemId, itemName, description, imageFile);
            
        } catch (Exception e) {
            log.error("이미지 URL에서 이미지 다운로드 실패: {}", imageUrl, e);
            // 이미지 다운로드 실패 시 텍스트만으로 임베딩 생성 시도
            try {
                createEmbedding(itemId, itemName, description, null);
            } catch (Exception ex) {
                log.warn("텍스트만으로 임베딩 생성도 실패: itemId={}", itemId, ex);
            }
        }
    }

    /**
     * Flask AI 서버에 배치 임베딩 생성 요청 (성능 최적화)
     * 여러 아이템을 한 번에 처리하여 네트워크 오버헤드 감소
     */
    public void createEmbeddingsBatch(List<LostItem> items) {
        if (items == null || items.isEmpty()) {
            return;
        }
        
        try {
            // 배치 데이터 구성
            List<Map<String, Object>> itemsData = new ArrayList<>();
            for (LostItem item : items) {
                Map<String, Object> itemData = new HashMap<>();
                itemData.put("item_id", item.getId());
                itemData.put("item_name", item.getItemName() != null ? item.getItemName() : "");  // 분실물 제목 추가
                itemData.put("description", item.getDescription() != null ? item.getDescription() : "");
                itemData.put("image_url", item.getImageUrl() != null ? item.getImageUrl() : "");
                itemsData.add(itemData);
            }
            
            // JSON으로 변환
            ObjectMapper objectMapper = new ObjectMapper();
            String itemsJson = objectMapper.writeValueAsString(itemsData);
            
            // 요청 본문 구성
            MultiValueMap<String, String> formData = new LinkedMultiValueMap<>();
            formData.add("items", itemsJson);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            
            HttpEntity<MultiValueMap<String, String>> requestEntity = new HttpEntity<>(formData, headers);
            
            Map<String, Object> response = flaskRestClient.post()
                    .uri("/api/v1/embedding/create-batch")
                    .body(requestEntity)
                    .retrieve()
                    .body(Map.class);
            
            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                Map<String, Object> summary = (Map<String, Object>) response.get("summary");
                if (summary != null) {
                    Integer successful = (Integer) summary.get("successful");
                    Integer failed = (Integer) summary.get("failed");
                    log.info("배치 임베딩 생성 완료: {}개 성공, {}개 실패", successful, failed);
                } else {
                    log.info("배치 임베딩 생성 완료: {}개 아이템", items.size());
                }
                return;
            }
            
            throw new RuntimeException("Failed to create batch embeddings: " + response);
            
        } catch (JsonProcessingException e) {
            log.error("배치 데이터 JSON 변환 실패", e);
            throw new RuntimeException("Failed to create batch embeddings", e);
        } catch (Exception e) {
            log.warn("Flask AI 서버 배치 API 호출 실패: {}", e.getMessage());
            throw new RuntimeException("Failed to create batch embeddings", e);
        }
    }

    /**
     * Flask AI 서버에 임베딩 삭제 요청
     */
    public void deleteEmbedding(Long itemId) {
        try {
            Map<String, Object> response = flaskRestClient.delete()
                    .uri("/api/v1/embedding/delete/" + itemId)
                    .retrieve()
                    .body(Map.class);

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Successfully deleted embedding for item {}", itemId);
                return;
            }
            
            log.warn("Failed to delete embedding for item {}: {}", itemId, response);
            
        } catch (Exception e) {
            log.warn("Flask AI 서버에 연결할 수 없습니다. 임베딩 삭제를 건너뜁니다. itemId: {}", itemId);
            // Flask 서버가 꺼져있어도 계속 진행
        }
    }

    /**
     * Flask AI 서버에 FAISS와 DB 동기화 요청 (Admin API)
     * DB에는 없지만 FAISS에는 있는 항목들을 삭제합니다.
     */
    public Map<String, Object> syncFaissWithDb(List<Long> dbItemIds) {
        try {
            Map<String, Object> request = Map.of("db_item_ids", dbItemIds);
            
            Map<String, Object> response = flaskRestClient.post()
                    .uri("/api/v1/admin/sync-with-db")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(Map.class);

            if (response != null && Boolean.TRUE.equals(response.get("success"))) {
                log.info("Successfully synced FAISS with DB. Deleted {} orphaned items", 
                        response.get("deleted_count"));
                return response;
            }
            
            log.warn("Failed to sync FAISS with DB: {}", response);
            return response != null ? response : Map.of("success", false, "message", "Unknown error");
            
        } catch (Exception e) {
            log.error("Flask AI 서버에 연결할 수 없습니다. FAISS 동기화 실패", e);
            throw new RuntimeException("Failed to sync FAISS with DB: " + e.getMessage(), e);
        }
    }
}

