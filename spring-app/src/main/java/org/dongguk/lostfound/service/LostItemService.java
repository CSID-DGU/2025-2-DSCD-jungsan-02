package org.dongguk.lostfound.service;

import com.google.cloud.storage.BlobInfo;
import com.google.cloud.storage.Storage;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.domain.user.UserErrorCode;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class LostItemService {
    @Value("${cloud.storage.bucket}")
    private String BUCKET_NAME;
    private final Storage storage;
    private final RestClient flaskRestClient;
    public final UserRepository userRepository;
    public final LostItemRepository lostItemRepository;

    @Transactional
    public void diagnosisPlant(
            Long userId,
            MultipartFile image
    ) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(UserErrorCode.USER_NOT_FOUND));

        // 1. 파일을 Resource 로 변환
        Resource imageResource = null;
        try {
            imageResource = new ByteArrayResource(image.getBytes()) {
                @Override
                @NonNull
                public String getFilename() {
                    return image.getOriginalFilename();
                }
            };
        } catch (IOException e) {
            throw new RuntimeException("Multipart 변환 중 예외 발생");
        }

        // 2. multipart/form-data 요청 바디 생성
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", imageResource)
                .contentType(MediaType.APPLICATION_PDF);

        // 3. RestClient 로 전송 (응답 방식 아직 미구현)
        flaskRestClient.post()
                .uri("/health")
                .body(builder.build());

        String imageUrl = null;
        try {
            imageUrl = uploadImage(userId, image.getBytes(), image.getOriginalFilename());
        } catch (IOException e) {
            throw new RuntimeException("이미지 GCS 저장 중 예외 발생");
        }
    }

    private String uploadImage(
            Long lostItemId,
            byte[] image,
            String imageName
    ) {
        UUID uuid = UUID.randomUUID();
        String objectName = "lost" + lostItemId + "/" + imageName + uuid;

        BlobInfo blobInfo = BlobInfo.newBuilder(BUCKET_NAME, objectName)
                .setContentType(probeContentType(imageName))
                .build();
        storage.create(blobInfo, image);

        return String.format("https://storage.googleapis.com/%s/%s", BUCKET_NAME, objectName);
    }

    private String probeContentType(String name) {
        String ext = name.substring(name.lastIndexOf('.') + 1).toLowerCase();
        return switch (ext) {
            case "png" -> "image/png";
            case "jpg", "jpeg" -> "image/jpeg";
            case "gif" -> "image/gif";
            case "bmp" -> "image/bmp";
            case "webp" -> "image/webp";
            default -> "application/octet-stream";
        };
    }
}
