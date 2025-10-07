package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.lostitem.LostItem;
import org.springframework.data.jpa.repository.JpaRepository;

public interface LostItemRepository extends JpaRepository<LostItem, Long> {
}
