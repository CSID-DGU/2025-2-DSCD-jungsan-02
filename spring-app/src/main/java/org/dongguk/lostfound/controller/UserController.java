package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.annotation.UserId;
import org.dongguk.lostfound.dto.response.MyPageDto;
import org.dongguk.lostfound.dto.response.UserDto;
import org.dongguk.lostfound.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/users")
public class UserController {
    private final UserService userService;

    /**
     * 내 정보 조회
     */
    @GetMapping("/me")
    public ResponseEntity<UserDto> getMyInfo(
            @UserId Long userId
    ) {
        UserDto result = userService.getUserInfo(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * 마이페이지 조회
     */
    @GetMapping("/mypage")
    public ResponseEntity<MyPageDto> getMyPage(
            @UserId Long userId
    ) {
        MyPageDto result = userService.getMyPage(userId);
        return ResponseEntity.ok(result);
    }
}
