package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.service.FlaskApiService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * FAISS 관리자용 컨트롤러
 */
@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/admin/faiss")
public class FaissAdminController {
    private final FlaskApiService flaskApiService;
    private final LostItemRepository lostItemRepository;

    /**
     * FAISS와 DB 동기화
     * DB에는 없지만 FAISS에는 있는 항목들을 찾아서 FAISS에서 삭제합니다.
     * 
     * 예: POST /api/v1/admin/faiss/sync
     */
    @PostMapping("/sync")
    public ResponseEntity<Map<String, Object>> syncFaissWithDb() {
        try {
            log.info("=== FAISS와 DB 동기화 API 호출됨 ===");
            
            // DB에 있는 모든 item_id 조회
            log.info("DB에서 모든 item_id 조회 시작...");
            List<Long> dbItemIds = lostItemRepository.findAll()
                    .stream()
                    .map(item -> item.getId())
                    .collect(Collectors.toList());
            
            log.info("DB에 존재하는 항목 수: {}개", dbItemIds.size());
            
            if (dbItemIds.isEmpty()) {
                log.warn("DB에 항목이 없습니다. 동기화를 건너뜁니다.");
                return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "DB에 항목이 없습니다.",
                    "total_db_items", 0,
                    "total_faiss_items", 0,
                    "deleted_count", 0
                ));
            }
            
            // Flask AI 서버에 동기화 요청
            log.info("Flask AI 서버에 동기화 요청 전송 시작... (item_id 개수: {})", dbItemIds.size());
            Map<String, Object> result = flaskApiService.syncFaissWithDb(dbItemIds);
            
            log.info("=== FAISS와 DB 동기화 완료: {} ===", result);
            
            return ResponseEntity.ok(result);
            
        } catch (Exception e) {
            log.error("FAISS 동기화 중 예외 발생", e);
            return ResponseEntity.status(500).body(Map.of(
                "success", false,
                "message", "동기화 실패: " + e.getMessage(),
                "error", e.getClass().getSimpleName()
            ));
        }
    }

    /**
     * FAISS 복구 (DB의 모든 아이템을 Flask에 재전송)
     * FAISS 인덱스가 비어있을 때 DB의 모든 아이템을 Flask 서버에 재전송하여 임베딩을 재생성합니다.
     * 
     * 예: POST /api/v1/admin/faiss/recover
     */
    @PostMapping("/recover")
    public ResponseEntity<Map<String, Object>> recoverFaiss() {
        try {
            log.info("=== FAISS 복구 API 호출됨 ===");
            
            // DB에 있는 모든 LostItem 조회
            log.info("DB에서 모든 분실물 조회 시작...");
            List<LostItem> allItems = lostItemRepository.findAll();
            
            log.info("DB에 존재하는 분실물 수: {}개", allItems.size());
            
            if (allItems.isEmpty()) {
                log.warn("DB에 분실물이 없습니다. 복구를 건너뜁니다.");
                return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "DB에 분실물이 없습니다.",
                    "total_items", 0,
                    "recovered_count", 0
                ));
            }
            
            // 배치 크기 설정 (메모리 및 네트워크 부하 고려)
            int batchSize = 10;
            int totalBatches = (int) Math.ceil((double) allItems.size() / batchSize);
            int recoveredCount = 0;
            int failedCount = 0;
            
            log.info("Flask 서버에 배치로 재전송 시작... (총 {}개 배치)", totalBatches);
            
            // 배치 단위로 나누어 Flask 서버에 재전송
            for (int i = 0; i < allItems.size(); i += batchSize) {
                int end = Math.min(i + batchSize, allItems.size());
                List<LostItem> batch = allItems.subList(i, end);
                
                try {
                    log.info("배치 {}/{} 전송 중... ({}개 아이템)", 
                        (i / batchSize) + 1, totalBatches, batch.size());
                    
                    flaskApiService.createEmbeddingsBatch(batch);
                    recoveredCount += batch.size();
                    
                    log.info("배치 {}/{} 전송 완료: {}개 아이템", 
                        (i / batchSize) + 1, totalBatches, batch.size());
                    
                    // 배치 간 짧은 대기 (서버 부하 방지)
                    Thread.sleep(100);
                    
                } catch (Exception e) {
                    log.error("배치 {}/{} 전송 실패: {}", 
                        (i / batchSize) + 1, totalBatches, e.getMessage(), e);
                    failedCount += batch.size();
                }
            }
            
            log.info("=== FAISS 복구 완료: {}개 성공, {}개 실패 ===", recoveredCount, failedCount);
            
            return ResponseEntity.ok(Map.of(
                "success", true,
                "message", String.format("복구 완료: %d개 성공, %d개 실패", recoveredCount, failedCount),
                "total_items", allItems.size(),
                "recovered_count", recoveredCount,
                "failed_count", failedCount
            ));
            
        } catch (Exception e) {
            log.error("FAISS 복구 중 예외 발생", e);
            return ResponseEntity.status(500).body(Map.of(
                "success", false,
                "message", "복구 실패: " + e.getMessage(),
                "error", e.getClass().getSimpleName()
            ));
        }
    }
}

