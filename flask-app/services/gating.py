"""
판단/게이팅 단계: 구조적 속성 불일치 결과를 명시적으로 제거 또는 페널티 부여

이 모듈은 후보 항목들에 대해 카테고리 및 속성 충돌을 검사하고,
충돌 시 페널티를 부여하거나 제거합니다.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from services.query_understanding import QueryAttributes
from services.attribute_extraction import (
    extract_attributes_from_text,
    has_attribute_conflict,
)


@dataclass
class GatingResult:
    """게이팅 단계 결과"""
    item_id: int
    passed: bool  # 게이팅 통과 여부
    penalty_score: float  # 페널티 점수 (음수, 0.0이면 페널티 없음)
    reasons: List[str]  # 게이팅 실패/통과 이유
    attribute_bonus: float  # 속성 일치 보너스 점수 (양수)


# 게이팅 파라미터 (점수 조정 가능)
CATEGORY_MISMATCH_PENALTY = -50.0  # 카테고리 불일치 페널티
COLOR_CONFLICT_PENALTY = -30.0  # 색상 충돌 페널티
PATTERN_CONFLICT_PENALTY = -30.0  # 패턴 충돌 페널티

COLOR_MATCH_BONUS = 20.0  # 색상 일치 보너스
PATTERN_MATCH_BONUS = 15.0  # 패턴 일치 보너스
BRAND_MATCH_BONUS = 25.0  # 브랜드 일치 보너스

MIN_SCORE_TO_PASS = -40.0  # 이 점수 이상이면 통과 (너무 낮으면 제거)


def check_category_mismatch(
    query_category: Optional[str],
    item_category: Optional[str]
) -> Tuple[bool, float]:
    """
    카테고리 불일치 검사
    
    Args:
        query_category: 쿼리에서 추출한 카테고리
        item_category: 항목의 카테고리
    
    Returns:
        (불일치 여부, 페널티 점수)
    """
    if query_category is None or item_category is None:
        # 카테고리가 명시되지 않으면 검사하지 않음
        return False, 0.0
    
    if query_category != item_category:
        return True, CATEGORY_MISMATCH_PENALTY
    
    return False, 0.0


def check_attribute_conflicts(
    query_attrs: Dict[str, List[str]],
    item_attrs: Dict[str, List[str]]
) -> Tuple[float, List[str]]:
    """
    속성 충돌 검사 및 페널티 계산
    
    Returns:
        (총 페널티 점수, 충돌 이유 리스트)
    """
    penalty = 0.0
    reasons = []
    
    # 색상 충돌
    if has_attribute_conflict(query_attrs, item_attrs, 'color'):
        penalty += COLOR_CONFLICT_PENALTY
        reasons.append(f"색상 불일치: 쿼리={query_attrs.get('color')}, 항목={item_attrs.get('color')}")
    
    # 패턴 충돌
    if has_attribute_conflict(query_attrs, item_attrs, 'pattern'):
        penalty += PATTERN_CONFLICT_PENALTY
        reasons.append(f"패턴 불일치: 쿼리={query_attrs.get('pattern')}, 항목={item_attrs.get('pattern')}")
    
    return penalty, reasons


def calculate_attribute_bonus(
    query_attrs: Dict[str, List[str]],
    item_attrs: Dict[str, List[str]]
) -> Tuple[float, List[str]]:
    """
    속성 일치 보너스 계산
    
    Returns:
        (총 보너스 점수, 일치 이유 리스트)
    """
    bonus = 0.0
    reasons = []
    
    # 색상 일치
    query_colors = set(query_attrs.get('color', []))
    item_colors = set(item_attrs.get('color', []))
    if query_colors and item_colors:
        if query_colors & item_colors:
            bonus += COLOR_MATCH_BONUS
            matched_colors = query_colors & item_colors
            reasons.append(f"색상 일치: {matched_colors}")
    
    # 패턴 일치
    query_patterns = set(query_attrs.get('pattern', []))
    item_patterns = set(item_attrs.get('pattern', []))
    if query_patterns and item_patterns:
        if query_patterns & item_patterns:
            bonus += PATTERN_MATCH_BONUS
            matched_patterns = query_patterns & item_patterns
            reasons.append(f"패턴 일치: {matched_patterns}")
    
    # 브랜드 일치
    query_brands = set(query_attrs.get('brand', []))
    item_brands = set(item_attrs.get('brand', []))
    if query_brands and item_brands:
        if query_brands & item_brands:
            bonus += BRAND_MATCH_BONUS
            matched_brands = query_brands & item_brands
            reasons.append(f"브랜드 일치: {matched_brands}")
    
    return bonus, reasons


def gate_item(
    item_id: int,
    query_attrs: QueryAttributes,
    item_category: Optional[str],
    item_description: str,
    item_name: str = "",
    item_brand: Optional[str] = None
) -> GatingResult:
    """
    단일 항목에 대한 게이팅 판단
    
    중요: 카테고리 불일치 체크 제거 (임베딩이 이미 의미를 포착)
    구조화된 메타데이터(색상, 브랜드)만 체크
    
    Args:
        item_id: 항목 ID
        query_attrs: 쿼리 속성 (category는 항상 None)
        item_category: 항목 카테고리 (구조화된 데이터)
        item_description: 항목 설명
        item_name: 항목 이름 (선택적)
        item_brand: 항목 브랜드 (구조화된 데이터, 선택적)
    
    Returns:
        GatingResult 객체
    """
    reasons = []
    penalty = 0.0
    bonus = 0.0
    
    # 카테고리 불일치 검사 제거 - 임베딩이 이미 의미를 포착하므로 불필요
    # 카테고리는 임베딩 유사도로 자연스럽게 처리됨
    
    # 항목에서 속성 추출
    item_text = f"{item_name} {item_description}".strip()
    item_attrs = extract_attributes_from_text(item_text)
    
    # 브랜드 정보가 구조화된 데이터로 제공되면 활용
    if item_brand and item_brand.strip():
        item_brand_normalized = item_brand.strip().lower()
        # 브랜드 키워드 사전과 매칭 시도
        from services.query_understanding import BRAND_KEYWORDS
        normalized_brand = None
        for brand_key, variants in BRAND_KEYWORDS.items():
            for variant in variants:
                if variant.lower() == item_brand_normalized:
                    normalized_brand = brand_key
                    break
            if normalized_brand:
                break
        
        if normalized_brand:
            if 'brand' not in item_attrs:
                item_attrs['brand'] = []
            if normalized_brand not in item_attrs['brand']:
                item_attrs['brand'].append(normalized_brand)
    
    # 속성 충돌 검사 (색상, 패턴, 브랜드)
    attr_penalty, conflict_reasons = check_attribute_conflicts(
        query_attrs.attributes,
        item_attrs
    )
    penalty += attr_penalty
    reasons.extend(conflict_reasons)
    
    # 속성 일치 보너스 계산
    attr_bonus, match_reasons = calculate_attribute_bonus(
        query_attrs.attributes,
        item_attrs
    )
    bonus += attr_bonus
    reasons.extend(match_reasons)
    
    # 최종 점수 계산
    total_score = penalty + bonus
    
    # 통과 여부 판단
    passed = total_score >= MIN_SCORE_TO_PASS
    
    if passed and not reasons:
        reasons.append("게이팅 통과 (충돌 없음)")
    
    return GatingResult(
        item_id=item_id,
        passed=passed,
        penalty_score=penalty,
        reasons=reasons,
        attribute_bonus=bonus
    )


def gate_candidates(
    candidate_item_ids: List[int],
    query_attrs: QueryAttributes,
    item_metadata: Dict[int, Dict]  # {item_id: {'category': ..., 'description': ..., 'item_name': ..., 'brand': ...}}
) -> Tuple[List[int], Dict[int, GatingResult]]:
    """
    후보 항목들에 대한 게이팅 수행
    
    중요: 카테고리 불일치 체크 제거, 구조화된 메타데이터만 활용
    
    Args:
        candidate_item_ids: 후보 item_id 리스트
        query_attrs: 쿼리 속성 (category는 항상 None)
        item_metadata: 항목 메타데이터 딕셔너리 (구조화된 데이터)
    
    Returns:
        (통과한 item_id 리스트, 각 항목의 GatingResult 딕셔너리)
    """
    passed_items = []
    gating_results = {}
    
    for item_id in candidate_item_ids:
        if item_id not in item_metadata:
            # 메타데이터가 없으면 통과 (다음 단계에서 처리)
            passed_items.append(item_id)
            continue
        
        metadata = item_metadata[item_id]
        result = gate_item(
            item_id=item_id,
            query_attrs=query_attrs,
            item_category=metadata.get('category'),  # 구조화된 데이터 (참고용, 비교 안 함)
            item_description=metadata.get('description', ''),
            item_name=metadata.get('item_name', ''),
            item_brand=metadata.get('brand')  # 구조화된 브랜드 데이터
        )
        
        gating_results[item_id] = result
        
        if result.passed:
            passed_items.append(item_id)
    
    return passed_items, gating_results

