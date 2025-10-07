package org.dongguk.lostfound.domain.lostitem;

import org.dongguk.lostfound.domain.type.ItemCategory;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

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

    @Builder(access = AccessLevel.PRIVATE)
    private LostItem(String itemName,
                     ItemCategory category,
                     String description,
                     LocalDate foundDate,
                     String location,
                     String imageUrl,
                     Long embeddingId) {
        this.itemName = itemName;
        this.category = category;
        this.description = description;
        this.foundDate = foundDate;
        this.location = location;
        this.imageUrl = imageUrl;
        this.embeddingId = embeddingId;
    }

    public static LostItem create(String itemName,
                                  ItemCategory category,
                                  String description,
                                  LocalDate foundDate,
                                  String location,
                                  String imageUrl,
                                  Long embeddingId) {
        return LostItem.builder()
                .itemName(itemName)
                .category(category)
                .description(description)
                .foundDate(foundDate)
                .location(location)
                .imageUrl(imageUrl)
                .embeddingId(embeddingId)
                .build();
    }

    public void updateEmbeddingId(Long embeddingId) {
        this.embeddingId = embeddingId;
    }
}
