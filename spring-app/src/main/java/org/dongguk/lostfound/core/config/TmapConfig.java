package org.dongguk.lostfound.core.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

@Slf4j
@Configuration
public class TmapConfig {

    @Value("${tmap.api.key}")
    private String tmapApiKey;

    @Value("${tmap.api.url:https://apis.openapi.sk.com/tmap}")
    private String tmapApiUrl;

    @Bean
    public RestClient tmapRestClient() {
        // API 키가 제대로 로드되었는지 확인
        if (tmapApiKey == null || tmapApiKey.trim().isEmpty()) {
            log.error("⚠️ TMap API 키가 설정되지 않았습니다! tmap.api.key 속성을 확인하세요.");
        } else {
            log.info("✅ TMap API 키가 로드되었습니다. (길이: {}자, 앞 10자: {}...)", 
                    tmapApiKey.length(), tmapApiKey.substring(0, Math.min(10, tmapApiKey.length())));
        }
        
        log.info("TMap API URL: {}", tmapApiUrl);
        
        return RestClient.builder()
                .baseUrl(tmapApiUrl)
                .defaultHeader("appKey", tmapApiKey)
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    @Bean
    public String tmapApiKey() {
        return tmapApiKey;
    }
}

