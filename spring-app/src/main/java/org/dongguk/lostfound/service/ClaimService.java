package org.dongguk.lostfound.service;

import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.Storage;
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
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ClaimService {
    @Value("${cloud.storage.bucket}")
    private String BUCKET_NAME;
    private final Storage storage;
    private final ClaimRequestRepository claimRequestRepository;
    private final LostItemRepository lostItemRepository;
    private final UserRepository userRepository;
    private final NotificationService notificationService;

    /**
     * 회수 요청 생성
     */
    @Transactional
    public ClaimRequestDto createClaimRequest(Long userId, Long lostItemId, CreateClaimRequest request, MultipartFile image) {
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
        
        // 이미지 업로드 (있는 경우)
        String imageUrl = null;
        if (image != null && !image.isEmpty()) {
            try {
                imageUrl = uploadImage(lostItemId, userId, image.getBytes(), image.getOriginalFilename());
            } catch (IOException e) {
                log.error("Failed to upload claim image", e);
                throw new RuntimeException("이미지 업로드 실패");
            }
        }
        
        // 회수 요청 생성
        ClaimRequest claimRequest = ClaimRequest.create(lostItem, claimer, request.message(), imageUrl);
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

    /**
     * 회수 요청 이미지 업로드
     */
    private String uploadImage(Long lostItemId, Long userId, byte[] image, String imageName) {
        UUID uuid = UUID.randomUUID();
        String objectName = "claims/" + lostItemId + "/" + userId + "/" + imageName + "_" + uuid;

        BlobInfo blobInfo = BlobInfo.newBuilder(BUCKET_NAME, objectName)
                .setContentType(probeContentType(imageName))
                .build();
        storage.create(blobInfo, image);

        return String.format("https://storage.googleapis.com/%s/%s", BUCKET_NAME, objectName);
    }

    /**
     * 이미지 Content-Type 추출
     */
    private String probeContentType(String name) {
        String ext = name.substring(name.lastIndexOf('.') + 1).toLowerCase();
        return switch (ext) {
            case "png" -> "image/png";
            case "jpg", "jpeg" -> "image/jpeg";
            case "gif" -> "image/gif";
            case "bmp" -> "image/bmp";
            case "webp" -> "image/webp";
            default -> "application/octet-stream";
        };
    }
}

