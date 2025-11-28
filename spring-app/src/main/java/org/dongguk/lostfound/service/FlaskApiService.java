package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
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
     * Flask AI 서버에 임베딩 생성 요청 (이미지 URL 사용)
     * CSV 임포트된 데이터의 이미지 URL에서 이미지를 다운로드하여 임베딩 생성
     */
    public void createEmbeddingFromUrl(Long itemId, String description, String imageUrl) {
        if (imageUrl == null || imageUrl.isEmpty()) {
            // 이미지가 없으면 텍스트만으로 임베딩 생성
            createEmbedding(itemId, description, null);
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
                createEmbedding(itemId, description, null);
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
            createEmbedding(itemId, description, imageFile);
            
        } catch (Exception e) {
            log.error("이미지 URL에서 이미지 다운로드 실패: {}", imageUrl, e);
            // 이미지 다운로드 실패 시 텍스트만으로 임베딩 생성 시도
            try {
                createEmbedding(itemId, description, null);
            } catch (Exception ex) {
                log.warn("텍스트만으로 임베딩 생성도 실패: itemId={}", itemId, ex);
            }
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

