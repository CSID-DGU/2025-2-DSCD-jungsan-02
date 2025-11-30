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
            // TMap API 요청 본문 구성 (노트북 코드와 동일한 형식)
            Map<String, Object> requestBody = Map.of(
                    "startX", String.valueOf(startLon),      // 경도
                    "startY", String.valueOf(startLat),       // 위도
                    "endX", String.valueOf(endLon),          // 경도
                    "endY", String.valueOf(endLat),          // 위도
                    "reqCoordType", "WGS84GEO",
                    "resCoordType", "WGS84GEO",
                    "startName", "출발지",
                    "endName", "도착지",
                    "searchOption", "0"  // 0: 최적 경로
            );

            // TMap API 호출: POST /routes/pedestrian?version=1
            var responseEntity = tmapRestClient.post()
                    .uri("/routes/pedestrian?version=1")
                    .body(requestBody)
                    .retrieve()
                    .toEntity(Map.class);

            // 상태 코드 확인
            if (!responseEntity.getStatusCode().is2xxSuccessful()) {
                log.warn("TMAP API 호출 실패: status={}, start: ({}, {}), end: ({}, {})", 
                        responseEntity.getStatusCode(), startLat, startLon, endLat, endLon);
                return null;
            }

            Map<String, Object> response = responseEntity.getBody();
            if (response == null) {
                log.warn("TMAP API 응답이 null입니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            // 응답 파싱: features[0].properties.totalDistance, totalTime
            if (!response.containsKey("features")) {
                log.warn("TMAP API 응답에 'features' 키가 없습니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            java.util.List<?> features = (java.util.List<?>) response.get("features");
            if (features == null || features.isEmpty()) {
                log.warn("TMAP API 응답의 'features' 배열이 비어있습니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            Map<String, Object> firstFeature = (Map<String, Object>) features.get(0);
            if (!firstFeature.containsKey("properties")) {
                log.warn("TMAP API 응답의 feature에 'properties' 키가 없습니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            Map<String, Object> properties = (Map<String, Object>) firstFeature.get("properties");
            Object distanceObj = properties.get("totalDistance"); // 미터
            Object timeObj = properties.get("totalTime");         // 초

            if (distanceObj == null || timeObj == null) {
                log.warn("TMAP API 응답에 totalDistance 또는 totalTime이 없습니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            // 타입 변환 (Integer 또는 Long일 수 있음)
            Integer totalDistance = null;
            Integer totalTime = null;

            if (distanceObj instanceof Integer) {
                totalDistance = (Integer) distanceObj;
            } else if (distanceObj instanceof Long) {
                totalDistance = ((Long) distanceObj).intValue();
            } else if (distanceObj instanceof Number) {
                totalDistance = ((Number) distanceObj).intValue();
            }

            if (timeObj instanceof Integer) {
                totalTime = (Integer) timeObj;
            } else if (timeObj instanceof Long) {
                totalTime = ((Long) timeObj).intValue();
            } else if (timeObj instanceof Number) {
                totalTime = ((Number) timeObj).intValue();
            }

            if (totalDistance == null || totalTime == null) {
                log.warn("TMAP API 응답의 totalDistance 또는 totalTime을 정수로 변환할 수 없습니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                return null;
            }

            // 초를 분으로 변환 (소수점 첫째 자리까지)
            double minutes = Math.round(totalTime / 60.0 * 10) / 10.0;
            
            log.debug("TMAP API 성공: 거리={}m, 시간={}분, start: ({}, {}), end: ({}, {})", 
                    totalDistance, minutes, startLat, startLon, endLat, endLon);
            
            return new TmapRouteResult(totalDistance, minutes);
            
        } catch (org.springframework.web.client.HttpClientErrorException e) {
            // 4xx 에러 (400, 401, 429 등)
            if (e.getStatusCode().value() == 429) {
                log.error("TMAP API 쿼터 초과 (429). start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
            } else {
                log.error("TMAP API 클라이언트 에러 ({}): {}. start: ({}, {}), end: ({}, {})", 
                        e.getStatusCode(), e.getResponseBodyAsString(), startLat, startLon, endLat, endLon);
            }
            return null;
        } catch (Exception e) {
            log.error("TMAP API 호출 중 예외 발생. start: ({}, {}), end: ({}, {}), error: {}", 
                    startLat, startLon, endLat, endLon, e.getMessage(), e);
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

