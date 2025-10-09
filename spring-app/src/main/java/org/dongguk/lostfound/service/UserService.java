package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.domain.claim.ClaimRequest;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ClaimStatus;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.dto.response.ClaimRequestDto;
import org.dongguk.lostfound.dto.response.LostItemDto;
import org.dongguk.lostfound.dto.response.MyPageDto;
import org.dongguk.lostfound.dto.response.UserDto;
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
public class UserService {
    private final UserRepository userRepository;
    private final LostItemRepository lostItemRepository;
    private final ClaimRequestRepository claimRequestRepository;

    /**
     * 사용자 정보 조회
     */
    public UserDto getUserInfo(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

        return UserDto.from(user);
    }

    /**
     * 마이페이지 정보 조회
     */
    public MyPageDto getMyPage(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

        // 내가 등록한 분실물 목록 조회
        List<LostItem> myLostItems = lostItemRepository.findByUserIdOrderByCreatedAtDesc(userId);

        List<LostItemDto> lostItemDtos = myLostItems.stream()
                .map(LostItemDto::from)
                .toList();

        // 내가 받은 회수 요청 목록 조회 (내가 등록한 분실물에 대한)
        List<ClaimRequest> receivedClaims = claimRequestRepository.findByLostItemUserIdOrderByCreatedAtDesc(userId);
        
        List<ClaimRequestDto> receivedClaimDtos = receivedClaims.stream()
                .map(ClaimRequestDto::from)
                .toList();
        
        // 대기중인 회수 요청 개수
        long pendingCount = receivedClaims.stream()
                .filter(claim -> claim.getStatus() == ClaimStatus.PENDING)
                .count();

        return MyPageDto.builder()
                .user(UserDto.from(user))
                .lostItems(lostItemDtos)
                .totalCount(lostItemDtos.size())
                .receivedClaimRequests(receivedClaimDtos)
                .pendingClaimCount((int) pendingCount)
                .build();
    }
}
