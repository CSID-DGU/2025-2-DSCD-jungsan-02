package org.dongguk.lostfound.service;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.repository.LostItemRepository;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class LostItemService {
    public final LostItemRepository lostItemRepository;
}
