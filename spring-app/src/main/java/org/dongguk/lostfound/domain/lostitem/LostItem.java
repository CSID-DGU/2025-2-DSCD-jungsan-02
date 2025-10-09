package org.dongguk.lostfound.domain.lostitem;

import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.type.LostItemStatus;
import org.dongguk.lostfound.domain.user.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "lost_items")
public class LostItem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "item_name", nullable = false, length = 100)
    private String itemName;

    @Enumerated(EnumType.STRING)
    @Column(name = "category", nullable = false, length = 50)
    private ItemCategory category;

    @Column(name = "description", nullable = false, length = 1000)
    private String description;

    @Column(name = "found_date", nullable = false)
    private LocalDate foundDate;

    @Column(name = "location", nullable = false, length = 255)
    private String location;

    @Column(name = "image_url", length = 500)
    private String imageUrl;

    @Column(name = "embedding_id", unique = true)
    private Long embeddingId;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 50)
    private LostItemStatus status;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Builder(access = AccessLevel.PRIVATE)
    private LostItem(String itemName,
                     ItemCategory category,
                     String description,
                     LocalDate foundDate,
                     String location,
                     String imageUrl,
                     Long embeddingId,
                     LostItemStatus status,
                     User user,
                     LocalDateTime createdAt,
                     LocalDateTime updatedAt) {
        this.itemName = itemName;
        this.category = category;
        this.description = description;
        this.foundDate = foundDate;
        this.location = location;
        this.imageUrl = imageUrl;
        this.embeddingId = embeddingId;
        this.status = status;
        this.user = user;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public static LostItem create(String itemName,
                                  ItemCategory category,
                                  String description,
                                  LocalDate foundDate,
                                  String location,
                                  String imageUrl,
                                  Long embeddingId,
                                  User user) {
        return LostItem.builder()
                .itemName(itemName)
                .category(category)
                .description(description)
                .foundDate(foundDate)
                .location(location)
                .imageUrl(imageUrl)
                .embeddingId(embeddingId)
                .status(LostItemStatus.REGISTERED)
                .user(user)
                .createdAt(LocalDateTime.now())
                .build();
    }

    public void updateEmbeddingId(Long embeddingId) {
        this.embeddingId = embeddingId;
    }

    public void updateStatus(LostItemStatus status) {
        this.status = status;
        this.updatedAt = LocalDateTime.now();
    }
}
