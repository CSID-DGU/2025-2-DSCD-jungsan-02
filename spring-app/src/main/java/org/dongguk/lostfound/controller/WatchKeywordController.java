package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.annotation.UserId;
import org.dongguk.lostfound.dto.request.CreateWatchKeywordRequest;
import org.dongguk.lostfound.dto.response.WatchKeywordDto;
import org.dongguk.lostfound.service.WatchKeywordService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/watch-keywords")
public class WatchKeywordController {
    private final WatchKeywordService watchKeywordService;

    /**
     * 키워드 등록
     */
    @PostMapping
    public ResponseEntity<WatchKeywordDto> createWatchKeyword(
            @UserId Long userId,
            @RequestBody CreateWatchKeywordRequest request
    ) {
        WatchKeywordDto result = watchKeywordService.createWatchKeyword(userId, request);
        return ResponseEntity.ok(result);
    }

    /**
     * 사용자의 키워드 목록 조회 (활성화된 것만)
     */
    @GetMapping
    public ResponseEntity<List<WatchKeywordDto>> getWatchKeywords(
            @UserId Long userId
    ) {
        List<WatchKeywordDto> result = watchKeywordService.getWatchKeywords(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * 사용자의 모든 키워드 조회 (비활성화 포함)
     */
    @GetMapping("/all")
    public ResponseEntity<List<WatchKeywordDto>> getAllWatchKeywords(
            @UserId Long userId
    ) {
        List<WatchKeywordDto> result = watchKeywordService.getAllWatchKeywords(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * 키워드 삭제 (비활성화)
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteWatchKeyword(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        watchKeywordService.deleteWatchKeyword(userId, id);
        return ResponseEntity.ok().build();
    }

    /**
     * 키워드 재활성화
     */
    @PutMapping("/{id}/activate")
    public ResponseEntity<Void> activateWatchKeyword(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        watchKeywordService.activateWatchKeyword(userId, id);
        return ResponseEntity.ok().build();
    }
}

