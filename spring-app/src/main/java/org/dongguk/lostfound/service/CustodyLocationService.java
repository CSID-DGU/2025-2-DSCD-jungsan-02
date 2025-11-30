package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.domain.custody.CustodyLocation;
import org.dongguk.lostfound.dto.request.NearbyCustodyLocationRequest;
import org.dongguk.lostfound.dto.response.CustodyLocationDto;
import org.dongguk.lostfound.repository.CustodyLocationRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class CustodyLocationService {
    private final CustodyLocationRepository custodyLocationRepository;
    private final TmapApiService tmapApiService;

    /**
     * 모든 보관소 목록 조회 (드롭다운용)
     */
    @Transactional(readOnly = true)
    public List<CustodyLocationDto> getAllCustodyLocations() {
        List<CustodyLocation> locations = custodyLocationRepository.findAll();
        return locations.stream()
                .map(CustodyLocationDto::from)
                .collect(Collectors.toList());
    }

    /**
     * 사용자 위치 기준으로 가까운 보관소 TopK 조회 (도보 거리 기준)
     * 1차: 10km 반경 내 보관소 필터링
     * 2차: 도보 거리 계산 후 상위 5개 반환
     */
    @Transactional(readOnly = true)
    public List<CustodyLocationDto> findNearbyCustodyLocations(NearbyCustodyLocationRequest request) {
        Double userLat = request.latitude();
        Double userLon = request.longitude();
        Integer topK = request.topK() != null ? request.topK() : 5; // 기본값 5개

        log.info("가까운 보관소 검색 요청: lat={}, lon={}, topK={}", userLat, userLon, topK);

        // 모든 보관소 조회
        List<CustodyLocation> allLocations = custodyLocationRepository.findAll();

        if (allLocations.isEmpty()) {
            log.warn("보관소 데이터가 없습니다.");
            return new ArrayList<>();
        }

        // 1차 필터링: 10km 반경 내 보관소만 선별
        final double RADIUS_KM = 10.0;
        final double RADIUS_METERS = RADIUS_KM * 1000.0;
        
        List<CustodyLocation> nearbyLocations = allLocations.stream()
                .filter(location -> {
                    double distance = calculateHaversineDistance(
                            userLat, userLon,
                            location.getLatitude(), location.getLongitude()
                    );
                    return distance <= RADIUS_METERS;
                })
                .collect(Collectors.toList());

        log.info("10km 반경 내 보관소 개수: {}/{}", nearbyLocations.size(), allLocations.size());

        if (nearbyLocations.isEmpty()) {
            log.info("10km 반경 내 보관소가 없습니다.");
            return new ArrayList<>();
        }

        // 2차: 각 보관소까지의 도보 거리 계산 (TMap API 사용)
        // 최대 처리 개수 제한 (너무 많으면 시간이 오래 걸림)
        int maxProcessCount = Math.min(nearbyLocations.size(), 20); // 최대 20개만 처리
        List<CustodyLocation> locationsToProcess = nearbyLocations.subList(0, maxProcessCount);
        
        if (nearbyLocations.size() > maxProcessCount) {
            log.info("보관소가 많아서 {}개만 처리합니다. (전체: {}개)", maxProcessCount, nearbyLocations.size());
        }
        
        List<CustodyLocationDto> results = new ArrayList<>();
        int processedCount = 0;
        boolean quotaExceeded = false; // 쿼터 초과 플래그
        
        for (CustodyLocation location : locationsToProcess) {
            // 쿼터 초과가 발생했으면 더 이상 호출하지 않음
            if (quotaExceeded) {
                log.warn("TMap API 쿼터 초과로 인해 나머지 보관소는 직선 거리로 계산합니다.");
                // 나머지는 직선 거리로 계산
                double straightDistance = calculateHaversineDistance(
                        userLat, userLon,
                        location.getLatitude(), location.getLongitude()
                );
                int estimatedWalkingDistance = (int) (straightDistance * 1.3);
                double estimatedWalkingTime = estimatedWalkingDistance / 70.0;
                
                CustodyLocationDto dto = CustodyLocationDto.from(
                        location,
                        estimatedWalkingDistance,
                        estimatedWalkingTime
                );
                results.add(dto);
                continue;
            }
            
            processedCount++;
            log.debug("보관소 거리 계산 중: {}/{} - {}", processedCount, locationsToProcess.size(), location.getName());
            
            // API 호출 전에 쿼터 초과 상태 확인
            if (tmapApiService.isQuotaExceeded()) {
                quotaExceeded = true;
                log.warn("TMap API 쿼터 초과 상태입니다. 나머지 보관소는 직선 거리로 계산합니다.");
            }
            
            TmapApiService.TmapRouteResult routeResult = null;
            if (!quotaExceeded) {
                routeResult = tmapApiService.getWalkingDistance(
                        userLat, userLon,
                        location.getLatitude(), location.getLongitude()
                );
                
                // 호출 후 쿼터 초과 상태 확인
                if (tmapApiService.isQuotaExceeded()) {
                    quotaExceeded = true;
                    log.error("TMap API 쿼터 초과 감지. 나머지 보관소는 직선 거리로 계산합니다.");
                }
            }

            if (routeResult != null) {
                CustodyLocationDto dto = CustodyLocationDto.from(
                        location,
                        routeResult.getDistance(),
                        routeResult.getTime()
                );
                results.add(dto);
            } else {
                // TMAP API 실패 시 직선 거리로 대체 (하버사인 공식)
                double straightDistance = calculateHaversineDistance(
                        userLat, userLon,
                        location.getLatitude(), location.getLongitude()
                );
                // 직선 거리를 대략적인 도보 거리로 추정 (1.3배)
                int estimatedWalkingDistance = (int) (straightDistance * 1.3);
                double estimatedWalkingTime = estimatedWalkingDistance / 70.0; // 분당 70m 가정

                CustodyLocationDto dto = CustodyLocationDto.from(
                        location,
                        estimatedWalkingDistance,
                        estimatedWalkingTime
                );
                results.add(dto);
            }

            // 쿼터 초과가 발생했으면 즉시 루프 중단
            if (quotaExceeded) {
                log.warn("TMap API 쿼터 초과로 인해 보관소 거리 계산을 중단합니다. (처리된 개수: {}/{})", 
                        processedCount, locationsToProcess.size());
                break; // 루프 즉시 중단
            }

            // API 호출 제한 방지를 위한 딜레이 (0.5초)
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        // 도보 거리 기준으로 정렬하고 TopK 반환
        return results.stream()
                .sorted(Comparator.comparingInt(CustodyLocationDto::walkingDistance))
                .limit(topK)
                .collect(Collectors.toList());
    }

    /**
     * 장소명을 기반으로 가까운 보관소 검색
     * 1. TMap API로 장소명 -> 좌표 변환
     * 2. 해당 좌표 기준 10km 반경 내 보관소 필터링
     * 3. 도보 거리 계산 후 상위 5개 반환
     * 
     * @param placeName 장소명 (예: "강남역", "홍대입구역")
     * @param topK 반환할 개수 (기본값 5)
     * @return 가까운 보관소 목록
     */
    @Transactional(readOnly = true)
    public List<CustodyLocationDto> findNearbyCustodyLocationsByPlaceName(String placeName, Integer topK) {
        log.info("장소명 기반 보관소 검색 요청: placeName={}, topK={}", placeName, topK);
        
        // 1. TMap API로 장소명 -> 좌표 변환
        TmapApiService.TmapPlaceResult placeResult = tmapApiService.searchPlace(placeName);
        
        if (placeResult == null) {
            log.warn("장소명 '{}'에 대한 좌표를 찾을 수 없습니다.", placeName);
            return new ArrayList<>();
        }
        
        Double placeLat = placeResult.getLatitude();
        Double placeLon = placeResult.getLongitude();
        String foundPlaceName = placeResult.getName();
        
        log.info("장소 검색 성공: 입력='{}', 검색결과='{}', 좌표=({}, {})", 
                placeName, foundPlaceName, placeLat, placeLon);
        
        // 2. 좌표 기반으로 가까운 보관소 검색 (기존 메서드 재사용)
        NearbyCustodyLocationRequest request = new NearbyCustodyLocationRequest(
                placeLat, 
                placeLon, 
                topK != null ? topK : 5
        );
        
        return findNearbyCustodyLocations(request);
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
}

