package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.util.JwtUtil;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.dto.response.JwtTokensDto;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AuthService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    @Transactional
    public JwtTokensDto signUp(
            String loginId,
            String password
    ) {
        if (userRepository.existsByLoginId(loginId)) {
            throw CustomException.type(UserErrorCode.USER_CONFLICT);
        }

        String encodedPassword = passwordEncoder.encode(password);
        User user = userRepository.save(User.create(loginId, encodedPassword));

        return jwtUtil.generateTokens(user.getId());
    }

    @Transactional
    public JwtTokensDto signIn(
            String loginId,
            String password
    ) {
        User user = userRepository.findByLoginId(loginId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

        if (!passwordEncoder.matches(password, user.getPassword())) {
            throw CustomException.type(UserErrorCode.NOT_MATCH_PASSWORD);
        }

        return jwtUtil.generateTokens(user.getId());
    }

    @Transactional(readOnly = true)
    public void checkDuplicateId(String loginId) {
        if (userRepository.existsByLoginId(loginId)) {
            throw CustomException.type(UserErrorCode.USER_CONFLICT);
        }
    }
}