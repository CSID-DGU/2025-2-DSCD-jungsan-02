"""
속성 추출 유틸리티: DB 항목의 description에서 속성(색상, 패턴 등)을 추출

이 모듈은 LostItem의 description 필드에서 구조화된 속성을 추출합니다.
쿼리 이해 단계와 동일한 키워드 사전을 사용하여 일관성을 보장합니다.
"""

import re
from typing import Dict, List, Optional
from services.query_understanding import (
    COLOR_KEYWORDS,
    PATTERN_KEYWORDS,
    BRAND_KEYWORDS,
    extract_colors,
    extract_patterns,
    extract_brands,
)


def extract_attributes_from_text(text: str) -> Dict[str, List[str]]:
    """
    텍스트(description)에서 속성을 추출
    
    Args:
        text: 분실물 설명 텍스트
    
    Returns:
        속성 딕셔너리
        {
            'color': ['검은색', '빨간색'],
            'pattern': ['체크'],
            'brand': ['나이키']
        }
    """
    if not text or not text.strip():
        return {}
    
    attributes = {}
    
    # 각 속성 추출
    colors = extract_colors(text)
    patterns = extract_patterns(text)
    brands = extract_brands(text)
    
    if colors:
        attributes['color'] = colors
    if patterns:
        attributes['pattern'] = patterns
    if brands:
        attributes['brand'] = brands
    
    return attributes


def has_attribute_conflict(
    query_attrs: Dict[str, List[str]],
    item_attrs: Dict[str, List[str]],
    attribute_type: str  # 'color', 'pattern', 'brand'
) -> bool:
    """
    쿼리 속성과 항목 속성 간 충돌 여부 확인
    
    Args:
        query_attrs: 쿼리에서 추출한 속성
        item_attrs: 항목에서 추출한 속성
        attribute_type: 확인할 속성 타입 ('color', 'pattern', 'brand')
    
    Returns:
        True: 충돌 (명시적으로 불일치)
        False: 충돌 없음 (일치하거나 명시되지 않음)
    
    Example:
        >>> query = {'color': ['검은색']}
        >>> item = {'color': ['흰색']}
        >>> has_attribute_conflict(query, item, 'color')
        True
    """
    query_values = query_attrs.get(attribute_type, [])
    item_values = item_attrs.get(attribute_type, [])
    
    # 둘 다 명시되지 않으면 충돌 없음
    if not query_values or not item_values:
        return False
    
    # 교집합이 있으면 충돌 없음 (일치)
    if set(query_values) & set(item_values):
        return False
    
    # 교집합이 없고 둘 다 명시되어 있으면 충돌
    return True


def calculate_attribute_match_score(
    query_attrs: Dict[str, List[str]],
    item_attrs: Dict[str, List[str]]
) -> float:
    """
    속성 일치 점수 계산 (0.0 ~ 1.0)
    
    점수 구성:
    - 색상 일치: 0.4
    - 패턴 일치: 0.3
    - 브랜드 일치: 0.3
    
    Args:
        query_attrs: 쿼리 속성
        item_attrs: 항목 속성
    
    Returns:
        0.0 ~ 1.0 사이의 점수
    """
    score = 0.0
    
    # 색상 일치 (가중치: 0.4)
    query_colors = set(query_attrs.get('color', []))
    item_colors = set(item_attrs.get('color', []))
    if query_colors and item_colors:
        if query_colors & item_colors:  # 교집합이 있으면 일치
            score += 0.4
    elif not query_colors or not item_colors:
        # 한쪽만 명시되지 않은 경우 점수 감소 없음 (중립)
        pass
    
    # 패턴 일치 (가중치: 0.3)
    query_patterns = set(query_attrs.get('pattern', []))
    item_patterns = set(item_attrs.get('pattern', []))
    if query_patterns and item_patterns:
        if query_patterns & item_patterns:
            score += 0.3
    elif not query_patterns or not item_patterns:
        pass
    
    # 브랜드 일치 (가중치: 0.3)
    query_brands = set(query_attrs.get('brand', []))
    item_brands = set(item_attrs.get('brand', []))
    if query_brands and item_brands:
        if query_brands & item_brands:
            score += 0.3
    elif not query_brands or not item_brands:
        pass
    
    return min(score, 1.0)


def calculate_keyword_overlap(query_keywords: List[str], item_text: str) -> float:
    """
    키워드 겹침 점수 계산 (Jaccard 유사도 기반)
    
    Args:
        query_keywords: 쿼리 키워드 리스트
        item_text: 항목 텍스트 (description + item_name)
    
    Returns:
        0.0 ~ 1.0 사이의 점수
    """
    if not query_keywords or not item_text:
        return 0.0
    
    # 항목 텍스트를 단어로 분리
    item_words = set(item_text.lower().split())
    query_words = set(kw.lower() for kw in query_keywords)
    
    # Jaccard 유사도
    intersection = query_words & item_words
    union = query_words | item_words
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

