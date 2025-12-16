package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.watchkeyword.WatchKeyword;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface WatchKeywordRepository extends JpaRepository<WatchKeyword, Long> {
    // 사용자별 키워드 조회 (활성화된 것만)
    List<WatchKeyword> findByUserIdAndIsActiveTrueOrderByCreatedAtDesc(Long userId);
    
    // 사용자별 모든 키워드 조회
    List<WatchKeyword> findByUserIdOrderByCreatedAtDesc(Long userId);
    
    // 활성화된 모든 키워드 조회 (분실물 등록 시 알림 발송용)
    List<WatchKeyword> findByIsActiveTrue();
    
    // 사용자별 특정 키워드 조회 (중복 체크용)
    Optional<WatchKeyword> findByUserIdAndKeyword(Long userId, String keyword);
}


