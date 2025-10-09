package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.notification.Notification;
import org.dongguk.lostfound.domain.type.NotificationStatus;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface NotificationRepository extends JpaRepository<Notification, Long> {
    // 사용자별 알림 조회 (최신순)
    List<Notification> findByUserIdAndStatusNotOrderByCreatedAtDesc(Long userId, NotificationStatus status);
    
    // 사용자별 + 타입별 알림 조회
    List<Notification> findByUserIdAndTypeAndStatusNotOrderByCreatedAtDesc(Long userId, NotificationType type, NotificationStatus status);
    
    // 사용자별 읽지 않은 알림 개수
    Long countByUserIdAndStatus(Long userId, NotificationStatus status);
}

