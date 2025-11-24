package org.dongguk.lostfound.core.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

@Configuration
public class TmapConfig {

    @Value("${tmap.api.key}")
    private String tmapApiKey;

    @Value("${tmap.api.url:https://apis.openapi.sk.com/tmap}")
    private String tmapApiUrl;

    @Bean
    public RestClient tmapRestClient() {
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

