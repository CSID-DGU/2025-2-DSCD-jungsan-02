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
import java.util.ArrayList;
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
            log.error("Error reading image file for item {}", itemId, e);
            throw new RuntimeException("Failed to create embedding", e);
        } catch (Exception e) {
            log.warn("Flask AI 서버에 연결할 수 없습니다. 임베딩 생성을 건너뜁니다. itemId: {}", itemId);
            // Flask 서버가 꺼져있어도 계속 진행
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
            log.warn("Flask AI 서버에 연결할 수 없습니다. 빈 결과를 반환합니다. query: {}", query);
            // Flask 서버가 꺼져있을 때 빈 리스트 반환 (fallback)
            return new ArrayList<>();
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
}

