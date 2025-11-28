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
        executor.setCorePoolSize(20); // 기본 스레드 수 (더 증가)
        executor.setMaxPoolSize(50); // 최대 스레드 수 (더 증가)
        executor.setQueueCapacity(1000); // 대기 큐 크기 (더 증가)
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

