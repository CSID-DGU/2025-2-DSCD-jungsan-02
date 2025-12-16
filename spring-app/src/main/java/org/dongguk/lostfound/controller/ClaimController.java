package org.dongguk.lostfound.controller;

import lombok.RequiredArgsConstructor;
import org.dongguk.lostfound.core.annotation.UserId;
import org.dongguk.lostfound.dto.request.CreateClaimRequest;
import org.dongguk.lostfound.dto.response.ClaimRequestDto;
import org.dongguk.lostfound.service.ClaimService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/claims")
public class ClaimController {
    private final ClaimService claimService;

    /**
     * 회수 요청 생성
     */
    @PostMapping(value = "/lost-items/{lostItemId}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ClaimRequestDto> createClaimRequest(
            @UserId Long userId,
            @PathVariable Long lostItemId,
            @RequestParam String message,
            @RequestParam(required = false) MultipartFile image
    ) {
        CreateClaimRequest request = new CreateClaimRequest(message);
        ClaimRequestDto result = claimService.createClaimRequest(userId, lostItemId, request, image);
        return ResponseEntity.ok(result);
    }

    /**
     * 회수 요청 승인
     */
    @PutMapping("/{id}/approve")
    public ResponseEntity<ClaimRequestDto> approveClaimRequest(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        ClaimRequestDto result = claimService.approveClaimRequest(userId, id);
        return ResponseEntity.ok(result);
    }

    /**
     * 회수 요청 거절
     */
    @PutMapping("/{id}/reject")
    public ResponseEntity<ClaimRequestDto> rejectClaimRequest(
            @UserId Long userId,
            @PathVariable Long id
    ) {
        ClaimRequestDto result = claimService.rejectClaimRequest(userId, id);
        return ResponseEntity.ok(result);
    }

    /**
     * 분실물에 대한 회수 요청 목록
     */
    @GetMapping("/lost-items/{lostItemId}")
    public ResponseEntity<List<ClaimRequestDto>> getClaimRequestsByLostItem(
            @PathVariable Long lostItemId
    ) {
        List<ClaimRequestDto> result = claimService.getClaimRequestsByLostItem(lostItemId);
        return ResponseEntity.ok(result);
    }

    /**
     * 내가 받은 회수 요청 목록
     */
    @GetMapping("/received")
    public ResponseEntity<List<ClaimRequestDto>> getReceivedClaimRequests(
            @UserId Long userId
    ) {
        List<ClaimRequestDto> result = claimService.getReceivedClaimRequests(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * 내가 보낸 회수 요청 목록
     */
    @GetMapping("/sent")
    public ResponseEntity<List<ClaimRequestDto>> getSentClaimRequests(
            @UserId Long userId
    ) {
        List<ClaimRequestDto> result = claimService.getSentClaimRequests(userId);
        return ResponseEntity.ok(result);
    }
}

