package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.domain.custody.CustodyLocation;
import org.dongguk.lostfound.repository.CustodyLocationRepository;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 보관장소 데이터 초기화 서비스
 * CSV 파일을 읽어서 데이터베이스에 저장
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class CustodyLocationInitService {
    private final CustodyLocationRepository custodyLocationRepository;

    /**
     * CSV 파일에서 보관장소 데이터를 읽어서 DB에 저장
     * application.yml에서 csv.file.path 설정 필요
     */
    @Transactional
    public void initializeCustodyLocations(String csvFilePath) {
        log.info("보관장소 데이터 초기화 시작: {}", csvFilePath);

        try {
            ClassPathResource resource = new ClassPathResource(csvFilePath);
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8)
            );

            String line;
            boolean isFirstLine = true;
            List<CustodyLocation> locations = new ArrayList<>();
            int savedCount = 0;
            int skippedCount = 0;

            while ((line = reader.readLine()) != null) {
                // 헤더 스킵
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                // CSV 파싱 (보관장소,분실물개수,위도,경도)
                String[] parts = line.split(",");
                if (parts.length < 4) {
                    skippedCount++;
                    continue;
                }

                try {
                    String name = parts[0].trim();
                    Integer itemCount = Integer.parseInt(parts[1].trim());
                    Double latitude = Double.parseDouble(parts[2].trim());
                    Double longitude = Double.parseDouble(parts[3].trim());

                    // 이미 존재하는지 확인
                    if (custodyLocationRepository.findByName(name).isEmpty()) {
                        CustodyLocation location = CustodyLocation.create(name, latitude, longitude, itemCount);
                        locations.add(location);
                        savedCount++;
                    } else {
                        skippedCount++;
                    }
                } catch (NumberFormatException e) {
                    log.warn("CSV 파싱 오류 (숫자 형식): {}", line);
                    skippedCount++;
                }
            }

            reader.close();

            // 배치 저장
            if (!locations.isEmpty()) {
                custodyLocationRepository.saveAll(locations);
                log.info("보관장소 데이터 초기화 완료: {}개 저장, {}개 스킵", savedCount, skippedCount);
            } else {
                log.info("저장할 새로운 보관장소 데이터가 없습니다.");
            }

        } catch (Exception e) {
            log.error("보관장소 데이터 초기화 실패: {}", csvFilePath, e);
            throw new RuntimeException("보관장소 데이터 초기화 실패", e);
        }
    }

    /**
     * 리소스 경로에서 CSV 파일 읽기 (기본 경로: custody-locations.csv)
     */
    @Transactional
    public void initializeCustodyLocationsFromDefaultPath() {
        initializeCustodyLocations("custody-locations.csv");
    }
}

