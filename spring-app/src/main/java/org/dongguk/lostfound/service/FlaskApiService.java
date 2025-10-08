package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.dto.response.SearchResultDto;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class FlaskApiService {
    private final RestClient flaskRestClient;

    /**
     * Flask AI 서버에 임베딩 생성 요청
     */
    public void createEmbedding(Long itemId, String description, MultipartFile image) {
        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("item_id", itemId.toString());
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
            log.error("Error creating embedding for item {}", itemId, e);
            throw new RuntimeException("Failed to create embedding", e);
        }
    }

    /**
     * Flask AI 서버에 검색 요청 (item_id 리스트 반환)
     */
    public List<Long> searchSimilarItems(String query, Integer topK) {
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
                return itemIds.stream()
                        .map(Long::valueOf)
                        .toList();
            }
            
            throw new RuntimeException("Failed to search embeddings: " + response);
            
        } catch (Exception e) {
            log.error("Error searching embeddings with query: {}", query, e);
            throw new RuntimeException("Failed to search embeddings", e);
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
            log.error("Error deleting embedding for item {}", itemId, e);
            // 삭제 실패해도 에러를 던지지 않음 (선택적)
        }
    }
}

