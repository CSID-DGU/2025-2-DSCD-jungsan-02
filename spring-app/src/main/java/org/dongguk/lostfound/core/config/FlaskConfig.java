package org.dongguk.lostfound.core.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

@Configuration
public class FlaskConfig {

    @Value("${flask.api.url}")
    private String flaskUrl; // http://flask-app:5001

    @Bean
    public RestClient flaskRestClient() {
        return RestClient.builder()
                .baseUrl(flaskUrl)
                .defaultHeader("Content-Type", MediaType.MULTIPART_FORM_DATA_VALUE)
                .build();
    }
}

