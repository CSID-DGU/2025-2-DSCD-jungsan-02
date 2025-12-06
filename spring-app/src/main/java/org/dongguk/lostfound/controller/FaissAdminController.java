package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.service.FlaskApiService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

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
        log.info("FAISS와 DB 동기화 시작");
        
        // DB에 있는 모든 item_id 조회
        List<Long> dbItemIds = lostItemRepository.findAll()
                .stream()
                .map(item -> item.getId())
                .collect(Collectors.toList());
        
        log.info("DB에 존재하는 항목 수: {}개", dbItemIds.size());
        
        // Flask AI 서버에 동기화 요청
        Map<String, Object> result = flaskApiService.syncFaissWithDb(dbItemIds);
        
        log.info("FAISS와 DB 동기화 완료: {}", result);
        
        return ResponseEntity.ok(result);
    }
}

