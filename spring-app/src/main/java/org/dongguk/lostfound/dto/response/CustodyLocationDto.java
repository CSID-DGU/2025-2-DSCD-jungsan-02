package org.dongguk.lostfound.dto.response;

import lombok.Builder;
import org.dongguk.lostfound.domain.custody.CustodyLocation;

@Builder
public record CustodyLocationDto(
        Long id,
        String name,
        Double latitude,
        Double longitude,
        Integer itemCount,
        Integer walkingDistance, // 미터
        Double walkingTime // 분
) {
    public static CustodyLocationDto from(CustodyLocation location) {
        return CustodyLocationDto.builder()
                .id(location.getId())
                .name(location.getName())
                .latitude(location.getLatitude())
                .longitude(location.getLongitude())
                .itemCount(location.getItemCount())
                .build();
    }

    public static CustodyLocationDto from(CustodyLocation location, Integer walkingDistance, Double walkingTime) {
        return CustodyLocationDto.builder()
                .id(location.getId())
                .name(location.getName())
                .latitude(location.getLatitude())
                .longitude(location.getLongitude())
                .itemCount(location.getItemCount())
                .walkingDistance(walkingDistance)
                .walkingTime(walkingTime)
                .build();
    }
}

