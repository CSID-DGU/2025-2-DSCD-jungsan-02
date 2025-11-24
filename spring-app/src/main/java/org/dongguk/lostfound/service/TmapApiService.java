package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class TmapApiService {
    private final RestClient tmapRestClient;

    /**
     * TMAP 도보 경로 API를 사용하여 두 좌표 간의 도보 거리와 시간을 계산
     * 
     * @param startLat 출발지 위도
     * @param startLon 출발지 경도
     * @param endLat 도착지 위도
     * @param endLon 도착지 경도
     * @return Map with "distance" (meters) and "time" (minutes), or null if failed
     */
    public TmapRouteResult getWalkingDistance(Double startLat, Double startLon, Double endLat, Double endLon) {
        try {
            Map<String, Object> requestBody = Map.of(
                    "startX", String.valueOf(startLon),
                    "startY", String.valueOf(startLat),
                    "endX", String.valueOf(endLon),
                    "endY", String.valueOf(endLat),
                    "reqCoordType", "WGS84GEO",
                    "resCoordType", "WGS84GEO",
                    "startName", "출발지",
                    "endName", "도착지",
                    "searchOption", "0"
            );

            Map<String, Object> response = tmapRestClient.post()
                    .uri("/routes/pedestrian?version=1")
                    .body(requestBody)
                    .retrieve()
                    .body(Map.class);

            if (response != null && response.containsKey("features") && 
                ((java.util.List<?>) response.get("features")).size() > 0) {
                
                java.util.List<?> features = (java.util.List<?>) response.get("features");
                Map<String, Object> properties = (Map<String, Object>) ((Map<String, Object>) features.get(0)).get("properties");
                
                Integer totalDistance = (Integer) properties.get("totalDistance"); // 미터
                Integer totalTime = (Integer) properties.get("totalTime"); // 초
                
                if (totalDistance != null && totalTime != null) {
                    return new TmapRouteResult(totalDistance, Math.round(totalTime / 60.0 * 10) / 10.0);
                }
            }
            
            log.warn("TMAP API 응답에서 거리/시간 정보를 찾을 수 없습니다. start: ({}, {}), end: ({}, {})", 
                    startLat, startLon, endLat, endLon);
            return null;
            
        } catch (Exception e) {
            log.error("TMAP API 호출 실패. start: ({}, {}), end: ({}, {})", 
                    startLat, startLon, endLat, endLon, e);
            return null;
        }
    }

    /**
     * TMAP API 결과를 담는 내부 클래스
     */
    public static class TmapRouteResult {
        private final Integer distance; // 미터
        private final Double time; // 분

        public TmapRouteResult(Integer distance, Double time) {
            this.distance = distance;
            this.time = time;
        }

        public Integer getDistance() {
            return distance;
        }

        public Double getTime() {
            return time;
        }
    }
}

