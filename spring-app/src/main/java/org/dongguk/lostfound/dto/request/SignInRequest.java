package org.dongguk.lostfound.dto.request;

public record SignInRequest(
        String loginId,
        String password
) {
}
