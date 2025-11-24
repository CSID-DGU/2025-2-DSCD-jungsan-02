package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.dongguk.lostfound.domain.type.LostItemStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

public interface LostItemRepository extends JpaRepository<LostItem, Long>, JpaSpecificationExecutor<LostItem> {
    Page<LostItem> findByCategory(ItemCategory category, Pageable pageable);
    Page<LostItem> findByFoundDateBetween(LocalDate startDate, LocalDate endDate, Pageable pageable);
    Page<LostItem> findByLocation(String location, Pageable pageable);

    // 사용자별 분실물 조회
    List<LostItem> findByUserIdOrderByCreatedAtDesc(Long userId);

    // 사용자별 + 상태별 분실물 조회
    List<LostItem> findByUserIdAndStatusOrderByCreatedAtDesc(Long userId, LostItemStatus status);

    // 오늘 등록된 분실물 개수
    Long countByCreatedAtAfter(LocalDateTime startOfDay);
    
    // 상태별 분실물 개수
    Long countByStatus(LostItemStatus status);
}
