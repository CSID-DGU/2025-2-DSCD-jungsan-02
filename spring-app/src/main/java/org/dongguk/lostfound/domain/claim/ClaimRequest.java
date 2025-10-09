package org.dongguk.lostfound.domain.claim;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ClaimStatus;
import org.dongguk.lostfound.domain.user.User;

import java.time.LocalDateTime;

/**
 * 분실물 회수 요청 엔티티
 */
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "claim_requests")
public class ClaimRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "lost_item_id", nullable = false)
    private LostItem lostItem;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "claimer_id", nullable = false)
    private User claimer;  // 회수 요청자 (분실자)

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 50)
    private ClaimStatus status;

    @Column(name = "message", length = 1000)
    private String message;  // 회수 요청 메시지

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Builder(access = AccessLevel.PRIVATE)
    private ClaimRequest(LostItem lostItem,
                        User claimer,
                        ClaimStatus status,
                        String message,
                        LocalDateTime createdAt,
                        LocalDateTime updatedAt) {
        this.lostItem = lostItem;
        this.claimer = claimer;
        this.status = status;
        this.message = message;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public static ClaimRequest create(LostItem lostItem, User claimer, String message) {
        return ClaimRequest.builder()
                .lostItem(lostItem)
                .claimer(claimer)
                .status(ClaimStatus.PENDING)
                .message(message)
                .createdAt(LocalDateTime.now())
                .build();
    }

    public void approve() {
        this.status = ClaimStatus.APPROVED;
        this.updatedAt = LocalDateTime.now();
    }

    public void reject() {
        this.status = ClaimStatus.REJECTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void complete() {
        this.status = ClaimStatus.COMPLETED;
        this.updatedAt = LocalDateTime.now();
    }
}

