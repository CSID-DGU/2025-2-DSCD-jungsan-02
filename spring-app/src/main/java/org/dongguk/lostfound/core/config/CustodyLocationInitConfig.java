package org.dongguk.lostfound.core.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.service.CustodyLocationInitService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
@RequiredArgsConstructor
public class CustodyLocationInitConfig {
    private final CustodyLocationInitService custodyLocationInitService;

    @Value("${custody.location.csv.path:}")
    private String csvPath;

    @Value("${custody.location.auto-init:false}")
    private boolean autoInit;

    /**
     * 애플리케이션 시작 시 보관장소 데이터 자동 초기화 (옵션)
     */
    @Bean
    public CommandLineRunner initCustodyLocations() {
        return args -> {
            if (autoInit) {
                if (csvPath != null && !csvPath.isEmpty()) {
                    log.info("보관장소 데이터 자동 초기화 시작...");
                    try {
                        custodyLocationInitService.initializeCustodyLocations(csvPath);
                    } catch (Exception e) {
                        log.error("보관장소 데이터 자동 초기화 실패", e);
                    }
                } else {
                    log.warn("custody.location.csv.path가 설정되지 않아 자동 초기화를 건너뜁니다.");
                }
            } else {
                log.info("보관장소 데이터 자동 초기화가 비활성화되어 있습니다. (custody.location.auto-init=false)");
            }
        };
    }
}

