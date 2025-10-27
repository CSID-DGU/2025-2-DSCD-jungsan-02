package org.dongguk.lostfound.core.constant;

public class AuthConstant {
    public static final String USER_ID_CLAIM_NAME = "uid";
    public static final String BEARER_PREFIX = "Bearer ";
    public static final String AUTHORIZATION_HEADER = "Authorization";
    public static final String ANONYMOUS_USER = "anonymousUser";
    public static final String[] AUTH_WHITELIST = {
            // 인증 API
            "/api/v1/auth/sign-up",
            "/api/v1/auth/sign-in",
            "/api/v1/auth/check-id",
            // 분실물 조회 API (로그인 필수 아님)
            "/api/v1/lost-items",
            "/api/v1/statistics"
    };
    private AuthConstant() {
    }
}