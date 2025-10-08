package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.user.User;

@Builder
public record UserDto(
        Long id,
        String loginId
) {
    public static UserDto from(User user) {
        return UserDto.builder()
                .id(user.getId())
                .loginId(user.getLoginId())
                .build();
    }
}

