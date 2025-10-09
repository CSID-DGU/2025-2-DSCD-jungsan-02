package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.notification.Notification;
import org.dongguk.lostfound.domain.type.NotificationStatus;
import org.dongguk.lostfound.domain.type.NotificationType;

import java.time.LocalDateTime;

@Builder
public record NotificationDto(
        Long id,
        NotificationType type,
        NotificationStatus status,
        String title,
        String message,
        Long relatedItemId,
        String relatedItemName,
        String actionUrl,
        LocalDateTime timestamp
) {
    public static NotificationDto from(Notification notification) {
        return NotificationDto.builder()
                .id(notification.getId())
                .type(notification.getType())
                .status(notification.getStatus())
                .title(notification.getTitle())
                .message(notification.getMessage())
                .relatedItemId(notification.getRelatedItemId())
                .relatedItemName(notification.getRelatedItemName())
                .actionUrl(notification.getActionUrl())
                .timestamp(notification.getCreatedAt())
                .build();
    }
}

