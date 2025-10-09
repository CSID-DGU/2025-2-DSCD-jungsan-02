package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.exception.GlobalErrorCode;
import org.dongguk.lostfound.domain.claim.ClaimRequest;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.LostItemStatus;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.dto.request.CreateClaimRequest;
import org.dongguk.lostfound.dto.response.ClaimRequestDto;
import org.dongguk.lostfound.repository.ClaimRequestRepository;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ClaimService {
    private final ClaimRequestRepository claimRequestRepository;
    private final LostItemRepository lostItemRepository;
    private final UserRepository userRepository;
    private final NotificationService notificationService;

    /**
     * 회수 요청 생성
     */
    @Transactional
    public ClaimRequestDto createClaimRequest(Long userId, Long lostItemId, CreateClaimRequest request) {
        User claimer = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        LostItem lostItem = lostItemRepository.findById(lostItemId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 자기 자신의 분실물에는 회수 요청 불가
        if (lostItem.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.BAD_JSON);
        }
        
        // 이미 회수 요청이 있는지 확인
        claimRequestRepository.findByLostItemIdAndClaimerId(lostItemId, userId)
                .ifPresent(existing -> {
                    throw CustomException.type(GlobalErrorCode.ALREADY_EXISTS);
                });
        
        // 회수 요청 생성
        ClaimRequest claimRequest = ClaimRequest.create(lostItem, claimer, request.message());
        claimRequest = claimRequestRepository.save(claimRequest);
        
        // 상태는 변경하지 않음 (승인 시에만 MATCHED로 변경)
        
        // 습득자(분실물 등록자)에게 알림 전송
        notificationService.createNotification(
                lostItem.getUser().getId(),
                NotificationType.REQUEST,
                "회수 요청이 도착했습니다",
                claimer.getLoginId() + "님이 '" + lostItem.getItemName() + "'의 회수를 요청했습니다.",
                lostItem.getId(),
                lostItem.getItemName(),
                "/mypage?tab=claims" // 마이페이지의 "받은 회수 요청" 탭으로 직접 이동
        );
        
        log.info("회수 요청 생성: lostItemId={}, claimerId={}", lostItemId, userId);
        
        return ClaimRequestDto.from(claimRequest);
    }

    /**
     * 회수 요청 승인 (습득자가)
     */
    @Transactional
    public ClaimRequestDto approveClaimRequest(Long userId, Long claimRequestId) {
        ClaimRequest claimRequest = claimRequestRepository.findById(claimRequestId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 본인이 등록한 분실물의 회수 요청인지 확인
        if (!claimRequest.getLostItem().getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }
        
        claimRequest.approve();
        
        // 분실물 상태를 MATCHED로 변경 (승인됨)
        claimRequest.getLostItem().updateStatus(LostItemStatus.MATCHED);
        
        // 회수 요청자에게 알림 전송
        notificationService.createNotification(
                claimRequest.getClaimer().getId(),
                NotificationType.APPROVED,
                "회수 요청이 승인되었습니다",
                "'" + claimRequest.getLostItem().getItemName() + "'의 회수 요청이 승인되었습니다.",
                claimRequest.getLostItem().getId(),
                claimRequest.getLostItem().getItemName(),
                "/mypage?tab=sent-claims" // 마이페이지의 "보낸 회수 요청" 탭으로 이동
        );
        
        log.info("회수 요청 승인: claimRequestId={}", claimRequestId);
        return ClaimRequestDto.from(claimRequest);
    }

    /**
     * 회수 요청 거절 (습득자가)
     */
    @Transactional
    public ClaimRequestDto rejectClaimRequest(Long userId, Long claimRequestId) {
        ClaimRequest claimRequest = claimRequestRepository.findById(claimRequestId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 본인이 등록한 분실물의 회수 요청인지 확인
        if (!claimRequest.getLostItem().getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }
        
        claimRequest.reject();
        
        // 분실물 상태를 다시 REGISTERED로 변경
        claimRequest.getLostItem().updateStatus(LostItemStatus.REGISTERED);
        
        // 회수 요청자에게 알림 전송
        notificationService.createNotification(
                claimRequest.getClaimer().getId(),
                NotificationType.REJECTED,
                "회수 요청이 거절되었습니다",
                "'" + claimRequest.getLostItem().getItemName() + "'의 회수 요청이 거절되었습니다.",
                claimRequest.getLostItem().getId(),
                claimRequest.getLostItem().getItemName(),
                "/mypage?tab=sent-claims" // 마이페이지의 "보낸 회수 요청" 탭으로 이동
        );
        
        log.info("회수 요청 거절: claimRequestId={}", claimRequestId);
        return ClaimRequestDto.from(claimRequest);
    }

    /**
     * 분실물에 대한 회수 요청 목록 조회
     */
    public List<ClaimRequestDto> getClaimRequestsByLostItem(Long lostItemId) {
        List<ClaimRequest> claimRequests = claimRequestRepository
                .findByLostItemIdOrderByCreatedAtDesc(lostItemId);
        
        return claimRequests.stream()
                .map(ClaimRequestDto::from)
                .toList();
    }

    /**
     * 내가 받은 회수 요청 목록 (내가 등록한 분실물에 대한)
     */
    public List<ClaimRequestDto> getReceivedClaimRequests(Long userId) {
        List<ClaimRequest> claimRequests = claimRequestRepository
                .findByLostItemUserIdOrderByCreatedAtDesc(userId);
        
        return claimRequests.stream()
                .map(ClaimRequestDto::from)
                .toList();
    }

    /**
     * 내가 보낸 회수 요청 목록
     */
    public List<ClaimRequestDto> getSentClaimRequests(Long userId) {
        List<ClaimRequest> claimRequests = claimRequestRepository
                .findByClaimerIdOrderByCreatedAtDesc(userId);
        
        return claimRequests.stream()
                .map(ClaimRequestDto::from)
                .toList();
    }
}

