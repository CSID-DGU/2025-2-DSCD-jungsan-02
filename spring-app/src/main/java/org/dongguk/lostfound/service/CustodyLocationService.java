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
     */
    @Transactional(readOnly = true)
    public List<CustodyLocationDto> findNearbyCustodyLocations(NearbyCustodyLocationRequest request) {
        Double userLat = request.latitude();
        Double userLon = request.longitude();
        Integer topK = request.topK() != null ? request.topK() : 10;

        log.info("가까운 보관소 검색 요청: lat={}, lon={}, topK={}", userLat, userLon, topK);

        // 모든 보관소 조회
        List<CustodyLocation> allLocations = custodyLocationRepository.findAll();

        if (allLocations.isEmpty()) {
            log.warn("보관소 데이터가 없습니다.");
            return new ArrayList<>();
        }

        // 각 보관소까지의 도보 거리 계산
        List<CustodyLocationDto> results = new ArrayList<>();
        
        for (CustodyLocation location : allLocations) {
            TmapApiService.TmapRouteResult routeResult = tmapApiService.getWalkingDistance(
                    userLat, userLon,
                    location.getLatitude(), location.getLongitude()
            );

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

            // API 호출 제한 방지를 위한 딜레이 (0.5초)
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        // 거리 기준으로 정렬하고 TopK 반환
        return results.stream()
                .sorted(Comparator.comparingInt(CustodyLocationDto::walkingDistance))
                .limit(topK)
                .collect(Collectors.toList());
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

