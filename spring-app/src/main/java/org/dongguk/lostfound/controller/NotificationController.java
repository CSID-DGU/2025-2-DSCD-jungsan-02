package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.annotation.UserId;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.dongguk.lostfound.dto.response.NotificationDto;
import org.dongguk.lostfound.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/notifications")
public class NotificationController {
    private final NotificationService notificationService;

    /**
     * 알림 목록 조회
     */
    @GetMapping
    public ResponseEntity<List<NotificationDto>> getNotifications(
            @UserId Long userId,
            @RequestParam(required = false) NotificationType type
    ) {
        if (type != null) {
            return ResponseEntity.ok(notificationService.getNotificationsByType(userId, type));
        }
        return ResponseEntity.ok(notificationService.getNotifications(userId));
    }

    /**
     * 읽지 않은 알림 개수 조회
     */
    @GetMapping("/unread-count")
    public ResponseEntity<Map<String, Long>> getUnreadCount(@UserId Long userId) {
        Long count = notificationService.getUnreadCount(userId);
        return ResponseEntity.ok(Map.of("count", count));
    }

    /**
     * 알림 읽음 처리
     */
    @PutMapping("/{id}/read")
    public ResponseEntity<Void> markAsRead(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        notificationService.markAsRead(userId, id);
        return ResponseEntity.ok().build();
    }

    /**
     * 모든 알림 읽음 처리
     */
    @PutMapping("/read-all")
    public ResponseEntity<Void> markAllAsRead(@UserId Long userId) {
        notificationService.markAllAsRead(userId);
        return ResponseEntity.ok().build();
    }

    /**
     * 알림 아카이브 처리
     */
    @PutMapping("/{id}/archive")
    public ResponseEntity<Void> archiveNotification(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        notificationService.archiveNotification(userId, id);
        return ResponseEntity.ok().build();
    }

    /**
     * 알림 삭제
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteNotification(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        notificationService.deleteNotification(userId, id);
        return ResponseEntity.ok().build();
    }
}

