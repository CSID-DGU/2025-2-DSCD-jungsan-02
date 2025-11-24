package org.dongguk.lostfound.domain.custody;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "custody_locations")
public class CustodyLocation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "name", nullable = false, length = 100, unique = true)
    private String name;

    @Column(name = "latitude", nullable = false)
    private Double latitude;

    @Column(name = "longitude", nullable = false)
    private Double longitude;

    @Column(name = "item_count", nullable = false)
    private Integer itemCount;

    @Builder(access = AccessLevel.PRIVATE)
    private CustodyLocation(String name, Double latitude, Double longitude, Integer itemCount) {
        this.name = name;
        this.latitude = latitude;
        this.longitude = longitude;
        this.itemCount = itemCount;
    }

    public static CustodyLocation create(String name, Double latitude, Double longitude, Integer itemCount) {
        return CustodyLocation.builder()
                .name(name)
                .latitude(latitude)
                .longitude(longitude)
                .itemCount(itemCount != null ? itemCount : 0)
                .build();
    }

    public void updateItemCount(Integer itemCount) {
        this.itemCount = itemCount;
    }
}

