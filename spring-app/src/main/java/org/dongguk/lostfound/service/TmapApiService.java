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
    private volatile boolean quotaExceeded = false; // 쿼터 초과 플래그 (volatile로 동시성 보장)

    /**
     * TMAP 도보 경로 API를 사용하여 두 좌표 간의 도보 거리와 시간을 계산
     * 
     * @param startLat 출발지 위도
     * @param startLon 출발지 경도
     * @param endLat 도착지 위도
     * @param endLon 도착지 경도
     * @return Map with "distance" (meters) and "time" (minutes), or null if failed
     */
    /**
     * 쿼터 초과 상태 확인
     */
    public boolean isQuotaExceeded() {
        return quotaExceeded;
    }
    
    /**
     * 쿼터 초과 플래그 리셋 (필요시 사용)
     */
    public void resetQuotaExceeded() {
        quotaExceeded = false;
    }

    public TmapRouteResult getWalkingDistance(Double startLat, Double startLon, Double endLat, Double endLon) {
        // 쿼터 초과 상태면 즉시 null 반환 (불필요한 호출 방지)
        if (quotaExceeded) {
            log.warn("TMap API 쿼터 초과 상태입니다. 호출을 건너뜁니다.");
            return null;
        }
        
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
                    .onStatus(status -> status.value() == 429, (request, response) -> {
                        quotaExceeded = true; // 쿼터 초과 플래그 설정
                        log.error("TMAP API 쿼터 초과 (429) 감지 (onStatus). 이후 호출은 건너뜁니다.");
                    })
                    .toEntity(Map.class);

            // 상태 코드 확인
            if (!responseEntity.getStatusCode().is2xxSuccessful()) {
                if (responseEntity.getStatusCode().value() == 429) {
                    quotaExceeded = true; // 쿼터 초과 플래그 설정
                    log.error("TMAP API 쿼터 초과 (429) 감지. 이후 호출은 건너뜁니다. start: ({}, {}), end: ({}, {})", 
                            startLat, startLon, endLat, endLon);
                } else {
                    log.warn("TMAP API 호출 실패: status={}, start: ({}, {}), end: ({}, {})", 
                            responseEntity.getStatusCode(), startLat, startLon, endLat, endLon);
                }
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
                quotaExceeded = true; // 쿼터 초과 플래그 설정
                log.error("TMAP API 쿼터 초과 (429) 감지. 이후 호출은 건너뜁니다. start: ({}, {}), end: ({}, {})", 
                        startLat, startLon, endLat, endLon);
                // 429 에러는 즉시 반환 (더 이상 호출하지 않음)
                return null;
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
     * TMap 장소 검색 API를 사용하여 장소명으로 좌표 검색
     * 
     * @param placeName 장소명 (예: "강남역", "홍대입구역")
     * @return 좌표 정보 (위도, 경도), 실패 시 null
     */
    public TmapPlaceResult searchPlace(String placeName) {
        // 쿼터 초과 상태면 즉시 null 반환
        if (quotaExceeded) {
            log.warn("TMap API 쿼터 초과 상태입니다. 장소 검색을 건너뜁니다.");
            return null;
        }
        
        try {
            log.info("TMap 장소 검색 요청: placeName={}", placeName);

            // TMap Search API 호출: GET /search/poi?version=1
            var responseEntity = tmapRestClient.get()
                    .uri(uriBuilder -> uriBuilder
                            .path("/search/poi")
                            .queryParam("version", "1")
                            .queryParam("searchKeyword", placeName)
                            .queryParam("searchType", "all")
                            .queryParam("searchtypCd", "A")
                            .queryParam("reqCoordType", "WGS84GEO")
                            .queryParam("resCoordType", "WGS84GEO")
                            .queryParam("count", "1")  // 첫 번째 결과만
                            .build())
                    .retrieve()
                    .onStatus(status -> status.value() == 403, (request, response) -> {
                        log.error("TMap 장소 검색 API 인증 실패 (403). 응답 본문: {}", 
                                response.getBody() != null ? response.getBody().toString() : "null");
                        throw new org.springframework.web.client.HttpClientErrorException(
                                org.springframework.http.HttpStatus.FORBIDDEN,
                                "TMap API 인증 실패"
                        );
                    })
                    .onStatus(status -> status.value() == 429, (request, response) -> {
                        quotaExceeded = true; // 쿼터 초과 플래그 설정
                        log.error("TMap 장소 검색 API 쿼터 초과 (429) 감지. 이후 호출은 건너뜁니다.");
                        throw new org.springframework.web.client.HttpClientErrorException(
                                org.springframework.http.HttpStatus.TOO_MANY_REQUESTS,
                                "TMap API 쿼터 초과"
                        );
                    })
                    .toEntity(Map.class);

            // 상태 코드 확인
            if (!responseEntity.getStatusCode().is2xxSuccessful()) {
                log.warn("TMap 장소 검색 API 호출 실패: status={}, placeName={}", 
                        responseEntity.getStatusCode(), placeName);
                return null;
            }

            Map<String, Object> response = responseEntity.getBody();
            if (response == null) {
                log.warn("TMap 장소 검색 API 응답이 null입니다. placeName={}", placeName);
                return null;
            }

            // 응답 파싱: searchPoiInfo.pois.poi[0].frontLat, frontLon
            if (!response.containsKey("searchPoiInfo")) {
                log.warn("TMap 장소 검색 API 응답에 'searchPoiInfo' 키가 없습니다. placeName={}", placeName);
                return null;
            }

            Map<String, Object> searchPoiInfo = (Map<String, Object>) response.get("searchPoiInfo");
            if (!searchPoiInfo.containsKey("pois")) {
                log.warn("TMap 장소 검색 API 응답에 'pois' 키가 없습니다. placeName={}", placeName);
                return null;
            }

            Map<String, Object> pois = (Map<String, Object>) searchPoiInfo.get("pois");
            if (!pois.containsKey("poi")) {
                log.warn("TMap 장소 검색 API 응답에 'poi' 키가 없습니다. placeName={}", placeName);
                return null;
            }

            Object poiObj = pois.get("poi");
            java.util.List<?> poiList = null;
            
            if (poiObj instanceof java.util.List) {
                poiList = (java.util.List<?>) poiObj;
            } else if (poiObj instanceof Map) {
                // 단일 객체인 경우 리스트로 변환
                poiList = java.util.List.of(poiObj);
            }

            if (poiList == null || poiList.isEmpty()) {
                log.warn("TMap 장소 검색 결과가 없습니다. placeName={}", placeName);
                return null;
            }

            Map<String, Object> firstPoi = (Map<String, Object>) poiList.get(0);
            Object latObj = firstPoi.get("frontLat");
            Object lonObj = firstPoi.get("frontLon");
            Object nameObj = firstPoi.get("name");

            if (latObj == null || lonObj == null) {
                log.warn("TMap 장소 검색 결과에 좌표가 없습니다. placeName={}", placeName);
                return null;
            }

            // 좌표 변환 (String 또는 Number일 수 있음)
            Double latitude = null;
            Double longitude = null;

            if (latObj instanceof String) {
                latitude = Double.parseDouble((String) latObj);
            } else if (latObj instanceof Number) {
                latitude = ((Number) latObj).doubleValue();
            }

            if (lonObj instanceof String) {
                longitude = Double.parseDouble((String) lonObj);
            } else if (lonObj instanceof Number) {
                longitude = ((Number) lonObj).doubleValue();
            }

            if (latitude == null || longitude == null) {
                log.warn("TMap 장소 검색 결과의 좌표를 변환할 수 없습니다. placeName={}", placeName);
                return null;
            }

            String foundName = nameObj != null ? nameObj.toString() : placeName;

            log.info("TMap 장소 검색 성공: placeName={}, foundName={}, lat={}, lon={}", 
                    placeName, foundName, latitude, longitude);

            return new TmapPlaceResult(foundName, latitude, longitude);

        } catch (org.springframework.web.client.HttpClientErrorException e) {
            if (e.getStatusCode().value() == 403) {
                log.error("TMap 장소 검색 API 인증 실패 (403 FORBIDDEN). placeName={}", placeName);
                log.error("응답 본문: {}", e.getResponseBodyAsString());
                log.error("⚠️ TMap API 키가 올바르게 설정되었는지 확인하세요. application.yml의 tmap.api.key 속성을 확인하세요.");
            } else if (e.getStatusCode().value() == 429) {
                quotaExceeded = true; // 쿼터 초과 플래그 설정
                log.error("TMap 장소 검색 API 쿼터 초과 (429) 감지. 이후 호출은 건너뜁니다. placeName={}", placeName);
            } else {
                log.error("TMap 장소 검색 API 클라이언트 에러 ({}): {}. placeName={}", 
                        e.getStatusCode(), e.getResponseBodyAsString(), placeName);
            }
            return null;
        } catch (Exception e) {
            log.error("TMap 장소 검색 API 호출 중 예외 발생. placeName={}, error: {}", 
                    placeName, e.getMessage(), e);
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

    /**
     * TMap 장소 검색 결과를 담는 내부 클래스
     */
    public static class TmapPlaceResult {
        private final String name;
        private final Double latitude;
        private final Double longitude;

        public TmapPlaceResult(String name, Double latitude, Double longitude) {
            this.name = name;
            this.latitude = latitude;
            this.longitude = longitude;
        }

        public String getName() {
            return name;
        }

        public Double getLatitude() {
            return latitude;
        }

        public Double getLongitude() {
            return longitude;
        }
    }
}

