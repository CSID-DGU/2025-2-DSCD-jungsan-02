package org.dongguk.lostfound.repository;

import org.dongguk.lostfound.domain.custody.CustodyLocation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface CustodyLocationRepository extends JpaRepository<CustodyLocation, Long> {
    Optional<CustodyLocation> findByName(String name);
}

