package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.claim.ClaimRequest;
import org.dongguk.lostfound.domain.type.ClaimStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ClaimRequestRepository extends JpaRepository<ClaimRequest, Long> {
    // 분실물별 회수 요청 목록
    List<ClaimRequest> findByLostItemIdOrderByCreatedAtDesc(Long lostItemId);
    
    // 사용자가 보낸 회수 요청 목록
    List<ClaimRequest> findByClaimerIdOrderByCreatedAtDesc(Long claimerId);
    
    // 사용자가 받은 회수 요청 목록 (내가 등록한 분실물에 대한)
    List<ClaimRequest> findByLostItemUserIdOrderByCreatedAtDesc(Long ownerId);
    
    // 특정 분실물에 대한 특정 사용자의 회수 요청 존재 여부
    Optional<ClaimRequest> findByLostItemIdAndClaimerId(Long lostItemId, Long claimerId);
    
    // 상태별 회수 요청 개수
    Long countByLostItemUserIdAndStatus(Long ownerId, ClaimStatus status);
}

