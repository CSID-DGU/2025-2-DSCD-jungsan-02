package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.dto.request.NearbyCustodyLocationRequest;
import org.dongguk.lostfound.dto.response.CustodyLocationDto;
import org.dongguk.lostfound.service.CustodyLocationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/custody-locations")
public class CustodyLocationController {
    private final CustodyLocationService custodyLocationService;

    /**
     * 모든 보관소 목록 조회 (드롭다운용)
     */
    @GetMapping
    public ResponseEntity<List<CustodyLocationDto>> getAllCustodyLocations() {
        List<CustodyLocationDto> result = custodyLocationService.getAllCustodyLocations();
        return ResponseEntity.ok(result);
    }

    /**
     * 사용자 위치 기준 가까운 보관소 검색
     */
    @PostMapping("/nearby")
    public ResponseEntity<List<CustodyLocationDto>> findNearbyCustodyLocations(
            @RequestBody NearbyCustodyLocationRequest request
    ) {
        List<CustodyLocationDto> result = custodyLocationService.findNearbyCustodyLocations(request);
        return ResponseEntity.ok(result);
    }

    /**
     * 장소명 기반 가까운 보관소 검색
     * 예: GET /api/v1/custody-locations/nearby-by-place?placeName=강남역&topK=5
     */
    @GetMapping("/nearby-by-place")
    public ResponseEntity<List<CustodyLocationDto>> findNearbyCustodyLocationsByPlaceName(
            @RequestParam String placeName,
            @RequestParam(required = false, defaultValue = "5") Integer topK
    ) {
        List<CustodyLocationDto> result = custodyLocationService.findNearbyCustodyLocationsByPlaceName(placeName, topK);
        return ResponseEntity.ok(result);
    }
}

