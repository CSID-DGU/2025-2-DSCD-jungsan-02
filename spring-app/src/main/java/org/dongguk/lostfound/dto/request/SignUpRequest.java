package org.dongguk.lostfound.dto.request;

public record SignUpRequest(
        String loginId,
        String password
) {
}
