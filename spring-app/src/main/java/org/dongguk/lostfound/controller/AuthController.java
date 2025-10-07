package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.dto.request.CheckDuplicateIdRequest;
import org.dongguk.lostfound.dto.request.SignInRequest;
import org.dongguk.lostfound.dto.request.SignUpRequest;
import org.dongguk.lostfound.dto.response.JwtTokensDto;
import org.dongguk.lostfound.service.AuthService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/auth")
public class AuthController {
    private final AuthService authService;

    @PostMapping("/sign-up")
    public ResponseEntity<JwtTokensDto> signUp(
            @RequestBody SignUpRequest signUpRequest
    ) {
        return ResponseEntity.ok(authService.signUp(signUpRequest.loginId(), signUpRequest.password()));
    }

    @PostMapping("/sign-in")
    public ResponseEntity<JwtTokensDto> signIn(
            @RequestBody SignInRequest signInRequest
    ) {
        return ResponseEntity.ok(authService.signIn(signInRequest.loginId(), signInRequest.password()));
    }

    @PostMapping("/check-id")
    public ResponseEntity<Boolean> checkDuplicate(
            @RequestBody CheckDuplicateIdRequest request
    ) {
        authService.checkDuplicateId(request.loginId());

        return ResponseEntity.ok(Boolean.TRUE);
    }

}
