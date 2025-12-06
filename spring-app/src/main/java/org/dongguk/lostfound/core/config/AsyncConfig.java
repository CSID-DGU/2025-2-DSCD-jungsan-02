package org.dongguk.lostfound.core.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.ThreadPoolExecutor;

@Configuration
@EnableAsync
public class AsyncConfig {

    /**
     * 임베딩 생성을 위한 비동기 실행자
     * 병렬 처리를 위해 스레드 풀 설정
     */
    @Bean(name = "embeddingExecutor")
    public ThreadPoolTaskExecutor embeddingExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        // Flask 서버 워커 수(2개)를 고려하여 동시 요청 수 제한
        // 각 워커가 안정적으로 처리할 수 있는 수준으로 설정
        executor.setCorePoolSize(4); // 기본 스레드 수 (20 -> 4로 감소)
        executor.setMaxPoolSize(8); // 최대 스레드 수 (50 -> 8로 감소)
        executor.setQueueCapacity(100); // 대기 큐 크기 (1000 -> 100으로 감소)
        executor.setThreadNamePrefix("embedding-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(120); // 종료 대기 시간 증가
        
        // 거부된 작업을 호출자 스레드에서 실행 (큐가 가득 찰 때)
        // 이렇게 하면 TaskRejectedException이 발생하지 않음
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        
        executor.initialize();
        return executor;
    }
}

