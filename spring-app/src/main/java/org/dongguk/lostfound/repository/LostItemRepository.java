package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.dongguk.lostfound.domain.type.ItemCategory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;

public interface LostItemRepository extends JpaRepository<LostItem, Long> {
    Page<LostItem> findByCategory(ItemCategory category, Pageable pageable);
    Page<LostItem> findByFoundDateBetween(LocalDate startDate, LocalDate endDate, Pageable pageable);
    Page<LostItem> findByLocation(String location, Pageable pageable);
}
