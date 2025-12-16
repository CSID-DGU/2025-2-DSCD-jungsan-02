package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.exception.GlobalErrorCode;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.watchkeyword.WatchKeyword;
import org.dongguk.lostfound.dto.request.CreateWatchKeywordRequest;
import org.dongguk.lostfound.dto.response.WatchKeywordDto;
import org.dongguk.lostfound.repository.UserRepository;
import org.dongguk.lostfound.repository.WatchKeywordRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class WatchKeywordService {
    private final WatchKeywordRepository watchKeywordRepository;
    private final UserRepository userRepository;
    private final NotificationService notificationService;

    /**
     * 키워드 등록
     */
    @Transactional
    public WatchKeywordDto createWatchKeyword(Long userId, CreateWatchKeywordRequest request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));

        String keyword = request.keyword().trim();
        if (keyword.isEmpty()) {
            throw CustomException.type(GlobalErrorCode.BAD_JSON);
        }

        // 중복 체크 (같은 사용자가 같은 키워드를 이미 등록했는지)
        watchKeywordRepository.findByUserIdAndKeyword(userId, keyword)
                .ifPresent(existing -> {
                    // 이미 존재하는 키워드가 비활성화 상태면 활성화
                    if (!existing.getIsActive()) {
                        existing.activate();
                        watchKeywordRepository.save(existing);
                        return;
                    }
                    throw CustomException.type(GlobalErrorCode.ALREADY_EXISTS);
                });

        WatchKeyword watchKeyword = WatchKeyword.create(user, keyword);
        watchKeyword = watchKeywordRepository.save(watchKeyword);

        log.info("Watch keyword created: userId={}, keyword={}", userId, keyword);
        return WatchKeywordDto.from(watchKeyword);
    }

    /**
     * 사용자의 키워드 목록 조회 (활성화된 것만)
     */
    public List<WatchKeywordDto> getWatchKeywords(Long userId) {
        List<WatchKeyword> keywords = watchKeywordRepository
                .findByUserIdAndIsActiveTrueOrderByCreatedAtDesc(userId);

        return keywords.stream()
                .map(WatchKeywordDto::from)
                .toList();
    }

    /**
     * 사용자의 모든 키워드 조회 (비활성화 포함)
     */
    public List<WatchKeywordDto> getAllWatchKeywords(Long userId) {
        List<WatchKeyword> keywords = watchKeywordRepository
                .findByUserIdOrderByCreatedAtDesc(userId);

        return keywords.stream()
                .map(WatchKeywordDto::from)
                .toList();
    }

    /**
     * 키워드 삭제 (비활성화)
     */
    @Transactional
    public void deleteWatchKeyword(Long userId, Long keywordId) {
        WatchKeyword watchKeyword = watchKeywordRepository.findById(keywordId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));

        // 본인의 키워드인지 확인
        if (!watchKeyword.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }

        watchKeyword.deactivate();
        watchKeywordRepository.save(watchKeyword);

        log.info("Watch keyword deactivated: userId={}, keywordId={}", userId, keywordId);
    }

    /**
     * 키워드 재활성화
     */
    @Transactional
    public void activateWatchKeyword(Long userId, Long keywordId) {
        WatchKeyword watchKeyword = watchKeywordRepository.findById(keywordId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));

        // 본인의 키워드인지 확인
        if (!watchKeyword.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }

        watchKeyword.activate();
        watchKeywordRepository.save(watchKeyword);

        log.info("Watch keyword activated: userId={}, keywordId={}", userId, keywordId);
    }

    /**
     * 분실물 등록 시 키워드 매칭 및 알림 발송
     * LostItemService에서 호출됨
     */
    @Transactional
    public void checkAndNotifyMatchingKeywords(LostItem lostItem) {
        // 활성화된 모든 키워드 조회
        List<WatchKeyword> activeKeywords = watchKeywordRepository.findByIsActiveTrue();

        if (activeKeywords.isEmpty()) {
            return;
        }

        // 분실물 정보를 하나의 문자열로 결합 (검색용)
        String searchText = buildSearchText(lostItem);

        int notificationCount = 0;
        for (WatchKeyword watchKeyword : activeKeywords) {
            // 본인이 등록한 분실물에는 알림 발송하지 않음
            if (watchKeyword.getUser().getId().equals(lostItem.getUser().getId())) {
                continue;
            }

            // 키워드 매칭 체크
            if (isKeywordMatched(searchText, watchKeyword.getKeyword())) {
                try {
                    // 알림 발송
                    notificationService.createNotification(
                            watchKeyword.getUser().getId(),
                            NotificationType.KEYWORD_MATCH,
                            "키워드 알림",
                            String.format("'%s' 키워드와 관련된 분실물이 등록되었습니다: %s",
                                    watchKeyword.getKeyword(), lostItem.getItemName()),
                            lostItem.getId(),
                            lostItem.getItemName(),
                            "/lost-items/" + lostItem.getId()
                    );
                    notificationCount++;
                    log.debug("Keyword match notification sent: userId={}, keyword={}, itemId={}",
                            watchKeyword.getUser().getId(), watchKeyword.getKeyword(), lostItem.getId());
                } catch (Exception e) {
                    log.error("Failed to send keyword notification: userId={}, keyword={}, itemId={}",
                            watchKeyword.getUser().getId(), watchKeyword.getKeyword(), lostItem.getId(), e);
                }
            }
        }

        if (notificationCount > 0) {
            log.info("Sent {} keyword notifications for lost item: itemId={}", notificationCount, lostItem.getId());
        }
    }

    /**
     * 분실물 정보를 검색 가능한 텍스트로 변환
     */
    private String buildSearchText(LostItem lostItem) {
        StringBuilder sb = new StringBuilder();
        sb.append(lostItem.getItemName()).append(" ");
        sb.append(lostItem.getDescription()).append(" ");
        if (lostItem.getBrand() != null && !lostItem.getBrand().isEmpty()) {
            sb.append(lostItem.getBrand()).append(" ");
        }
        return sb.toString().toLowerCase();
    }

    /**
     * 키워드 매칭 체크
     * 간단한 문자열 포함 체크 (향후 AI 검색으로 확장 가능)
     */
    private boolean isKeywordMatched(String searchText, String keyword) {
        String lowerKeyword = keyword.toLowerCase().trim();
        return searchText.contains(lowerKeyword);
    }
}

