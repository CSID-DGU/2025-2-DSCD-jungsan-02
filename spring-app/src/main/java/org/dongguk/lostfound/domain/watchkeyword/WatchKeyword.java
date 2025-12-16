package org.dongguk.lostfound.domain.watchkeyword;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.dongguk.lostfound.domain.user.User;

import java.time.LocalDateTime;

/**
 * 키워드 감시 엔티티
 * 사용자가 등록한 키워드와 관련된 분실물이 등록되면 알림을 받습니다.
 */
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "watch_keywords")
public class WatchKeyword {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "keyword", nullable = false, length = 255)
    private String keyword;

    @Column(name = "is_active", nullable = false)
    private Boolean isActive;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Builder(access = AccessLevel.PRIVATE)
    private WatchKeyword(User user, String keyword, Boolean isActive, 
                        LocalDateTime createdAt, LocalDateTime updatedAt) {
        this.user = user;
        this.keyword = keyword;
        this.isActive = isActive;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public static WatchKeyword create(User user, String keyword) {
        return WatchKeyword.builder()
                .user(user)
                .keyword(keyword.trim())
                .isActive(true)
                .createdAt(LocalDateTime.now())
                .build();
    }

    public void deactivate() {
        this.isActive = false;
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.isActive = true;
        this.updatedAt = LocalDateTime.now();
    }
}


