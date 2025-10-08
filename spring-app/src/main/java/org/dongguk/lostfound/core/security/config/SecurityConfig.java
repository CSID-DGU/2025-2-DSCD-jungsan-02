package org.dongguk.lostfound.core.security.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.constant.AuthConstant;
import org.dongguk.lostfound.core.security.filter.JwtAuthenticationFilter;
import org.dongguk.lostfound.core.security.filter.JwtExceptionFilter;
import org.dongguk.lostfound.core.security.handler.JwtAuthenticationEntryPoint;
import org.dongguk.lostfound.core.util.JwtUtil;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.security.servlet.PathRequest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.Arrays;
import java.util.List;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    // prod에서는 false로 내려 Nginx만 CORS 담당하게 함
    @Value("${app.cors.enabled:true}")
    private boolean corsEnabled;

    // 쉼표로 구분된 오리진 목록 (없으면 기본 로컬 패턴 사용)
    @Value("${app.cors.allowed-origins:}")
    private String allowedOriginsProp;

    private final JwtUtil jwtUtil;
    private final JwtAuthenticationEntryPoint jwtAuthenticationEntryPoint;
    private final ObjectMapper objectMapper;

    @Bean
    public JwtExceptionFilter jwtExceptionFilter() {
        return new JwtExceptionFilter(objectMapper);
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    /**
     * 로컬/스테이징에서 사용할 CORS 설정
     * - allowCredentials(true) 상황에서는 '*'를 쓰지 말고 'allowedOriginPatterns'를 사용해야 함
     */
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration c = new CorsConfiguration();

        // 우선순위: 설정 파일에 명시된 오리진 > 기본 로컬 패턴
        if (allowedOriginsProp != null && !allowedOriginsProp.isBlank()) {
            List<String> origins = Arrays.stream(allowedOriginsProp.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isBlank())
                    .toList();
            origins.forEach(c::addAllowedOriginPattern);
        } else {
            // 기본 로컬 개발 환경 허용
            c.addAllowedOriginPattern("http://localhost:*");
            c.addAllowedOriginPattern("https://localhost:*");
            c.addAllowedOriginPattern("http://127.0.0.1:*");
        }

        c.addAllowedMethod("*");     // GET, POST, PUT, DELETE, PATCH, OPTIONS
        c.addAllowedHeader("*");     // Origin, Content-Type, Authorization, ...
        c.setAllowCredentials(true); // 쿠키/자격증명 허용

        UrlBasedCorsConfigurationSource s = new UrlBasedCorsConfigurationSource();
        s.registerCorsConfiguration("/**", c);
        return s;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(AbstractHttpConfigurer::disable)
                .httpBasic(AbstractHttpConfigurer::disable)
                .formLogin(AbstractHttpConfigurer::disable)
                .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        // 프리플라이트는 반드시 열어둔다
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                        .requestMatchers(AuthConstant.AUTH_WHITELIST).permitAll()
                        .requestMatchers(PathRequest.toStaticResources().atCommonLocations()).permitAll()
                        .anyRequest().authenticated()
                )
                .exceptionHandling(e -> e.authenticationEntryPoint(jwtAuthenticationEntryPoint));

        // CORS 토글: true면 Spring에서 CORS 허용, false면 비활성화(Nginx만 사용)
        if (corsEnabled) {
            http.cors(cors -> cors.configurationSource(corsConfigurationSource()));
        } else {
            http.cors(AbstractHttpConfigurer::disable);
        }

        return http
                .addFilterBefore(new JwtAuthenticationFilter(jwtUtil), UsernamePasswordAuthenticationFilter.class)
                .addFilterBefore(jwtExceptionFilter(), UsernamePasswordAuthenticationFilter.class)
                .build();
    }
}
