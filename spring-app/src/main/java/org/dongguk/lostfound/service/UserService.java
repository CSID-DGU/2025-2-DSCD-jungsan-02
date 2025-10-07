package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;
}
