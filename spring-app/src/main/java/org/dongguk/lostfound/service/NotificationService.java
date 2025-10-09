package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dongguk.lostfound.core.exception.CustomException;
import org.dongguk.lostfound.core.exception.GlobalErrorCode;
import org.dongguk.lostfound.domain.notification.Notification;
import org.dongguk.lostfound.domain.type.NotificationStatus;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.dongguk.lostfound.domain.user.User;
import org.dongguk.lostfound.dto.response.NotificationDto;
import org.dongguk.lostfound.repository.NotificationRepository;
import org.dongguk.lostfound.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class NotificationService {
    private final NotificationRepository notificationRepository;
    private final UserRepository userRepository;

    /**
     * 사용자의 모든 알림 조회 (아카이브 제외)
     */
    public List<NotificationDto> getNotifications(Long userId) {
        List<Notification> notifications = notificationRepository
                .findByUserIdAndStatusNotOrderByCreatedAtDesc(userId, NotificationStatus.ARCHIVED);
        
        return notifications.stream()
                .map(NotificationDto::from)
                .toList();
    }

    /**
     * 사용자의 특정 타입 알림 조회
     */
    public List<NotificationDto> getNotificationsByType(Long userId, NotificationType type) {
        List<Notification> notifications = notificationRepository
                .findByUserIdAndTypeAndStatusNotOrderByCreatedAtDesc(userId, type, NotificationStatus.ARCHIVED);
        
        return notifications.stream()
                .map(NotificationDto::from)
                .toList();
    }

    /**
     * 읽지 않은 알림 개수 조회
     */
    public Long getUnreadCount(Long userId) {
        return notificationRepository.countByUserIdAndStatus(userId, NotificationStatus.UNREAD);
    }

    /**
     * 알림 읽음 처리
     */
    @Transactional
    public void markAsRead(Long userId, Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 본인의 알림인지 확인
        if (!notification.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }
        
        notification.markAsRead();
    }

    /**
     * 모든 알림 읽음 처리
     */
    @Transactional
    public void markAllAsRead(Long userId) {
        List<Notification> notifications = notificationRepository
                .findByUserIdAndStatusNotOrderByCreatedAtDesc(userId, NotificationStatus.ARCHIVED);
        
        notifications.forEach(Notification::markAsRead);
    }

    /**
     * 알림 아카이브 처리
     */
    @Transactional
    public void archiveNotification(Long userId, Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 본인의 알림인지 확인
        if (!notification.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }
        
        notification.archive();
    }

    /**
     * 알림 삭제
     */
    @Transactional
    public void deleteNotification(Long userId, Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        // 본인의 알림인지 확인
        if (!notification.getUser().getId().equals(userId)) {
            throw CustomException.type(GlobalErrorCode.FORBIDDEN);
        }
        
        notificationRepository.delete(notification);
    }

    /**
     * 알림 생성 (내부 사용)
     */
    @Transactional
    public void createNotification(Long userId, NotificationType type, String title, String message,
                                   Long relatedItemId, String relatedItemName, String actionUrl) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> CustomException.type(GlobalErrorCode.NOT_FOUND));
        
        Notification notification = Notification.create(
                type, title, message, user,
                relatedItemId, relatedItemName, actionUrl
        );
        
        notificationRepository.save(notification);
    }
}

