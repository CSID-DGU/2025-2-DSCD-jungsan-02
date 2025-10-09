package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.dto.response.StatisticsDto;
import org.dongguk.lostfound.service.LostItemService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/statistics")
public class StatisticsController {
    private final LostItemService lostItemService;

    /**
     * 전체 통계 데이터 조회
     */
    @GetMapping
    public ResponseEntity<StatisticsDto> getStatistics() {
        StatisticsDto statistics = lostItemService.getStatistics();
        return ResponseEntity.ok(statistics);
    }
}

