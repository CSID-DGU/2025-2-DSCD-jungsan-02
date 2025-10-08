package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.dto.response.MyPageDto;
import org.dongguk.lostfound.dto.response.UserDto;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class UserService {
    private final UserRepository userRepository;

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
     * TODO: 내가 등록한 분실물 목록도 함께 반환
     */
    public MyPageDto getMyPage(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

        // TODO: 내가 등록한 분실물 목록 조회
        // 현재는 사용자 정보만 반환
        return MyPageDto.builder()
                .user(UserDto.from(user))
                .myLostItems(java.util.List.of())
                .totalCount(0)
                .build();
    }
}
