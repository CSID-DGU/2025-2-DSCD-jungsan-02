package org.dongguk.lostfound.domain.notification;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.dongguk.lostfound.domain.type.NotificationStatus;
import org.dongguk.lostfound.domain.type.NotificationType;
import org.dongguk.lostfound.domain.user.User;

import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "notifications")
public class Notification {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(name = "type", nullable = false, length = 50)
    private NotificationType type;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 50)
    private NotificationStatus status;

    @Column(name = "title", nullable = false, length = 200)
    private String title;

    @Column(name = "message", nullable = false, length = 1000)
    private String message;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "related_item_id")
    private Long relatedItemId;

    @Column(name = "related_item_name", length = 200)
    private String relatedItemName;

    @Column(name = "action_url", length = 500)
    private String actionUrl;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Builder(access = AccessLevel.PRIVATE)
    private Notification(NotificationType type,
                        NotificationStatus status,
                        String title,
                        String message,
                        User user,
                        Long relatedItemId,
                        String relatedItemName,
                        String actionUrl,
                        LocalDateTime createdAt) {
        this.type = type;
        this.status = status;
        this.title = title;
        this.message = message;
        this.user = user;
        this.relatedItemId = relatedItemId;
        this.relatedItemName = relatedItemName;
        this.actionUrl = actionUrl;
        this.createdAt = createdAt;
    }

    public static Notification create(NotificationType type,
                                     String title,
                                     String message,
                                     User user,
                                     Long relatedItemId,
                                     String relatedItemName,
                                     String actionUrl) {
        return Notification.builder()
                .type(type)
                .status(NotificationStatus.UNREAD)
                .title(title)
                .message(message)
                .user(user)
                .relatedItemId(relatedItemId)
                .relatedItemName(relatedItemName)
                .actionUrl(actionUrl)
                .createdAt(LocalDateTime.now())
                .build();
    }

    public void markAsRead() {
        this.status = NotificationStatus.READ;
    }

    public void archive() {
        this.status = NotificationStatus.ARCHIVED;
    }
}

