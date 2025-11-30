package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.service.CsvDataImportService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * CSV 데이터 임포트를 위한 관리자용 컨트롤러
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/admin/csv-import")
public class CsvImportController {
    private final CsvDataImportService csvDataImportService;

    /**
     * 보관소 위치 데이터 임포트
     * 예: POST /api/v1/admin/csv-import/custody-locations?filePath=/path/to/custody_locations_with_coords.csv
     */
    @PostMapping("/custody-locations")
    public ResponseEntity<CsvDataImportService.ImportResult> importCustodyLocations(
            @RequestParam String filePath
    ) {
        CsvDataImportService.ImportResult result = csvDataImportService.importCustodyLocations(filePath);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물 데이터 임포트
     * 예: POST /api/v1/admin/csv-import/lost-items?filePath=/path/to/251013_lost112_place.csv
     */
    @PostMapping("/lost-items")
    public ResponseEntity<CsvDataImportService.ImportResult> importLostItems(
            @RequestParam String filePath
    ) {
        CsvDataImportService.ImportResult result = csvDataImportService.importLostItems(filePath);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물 데이터 테스트 임포트 (제한된 개수만)
     * limit을 지정하지 않으면 전체 임포트, limit을 지정하면 해당 개수만큼만 임포트
     * 예: POST /api/v1/admin/csv-import/lost-items/test?filePath=/path/to/251013_lost112_place.csv&limit=1000
     */
    @PostMapping("/lost-items/test")
    public ResponseEntity<CsvDataImportService.ImportResult> importLostItemsTest(
            @RequestParam String filePath,
            @RequestParam(required = false) Integer limit
    ) {
        int actualLimit = (limit != null) ? limit : Integer.MAX_VALUE;
        CsvDataImportService.ImportResult result = csvDataImportService.importLostItems(filePath, actualLimit);
        return ResponseEntity.ok(result);
    }
}

