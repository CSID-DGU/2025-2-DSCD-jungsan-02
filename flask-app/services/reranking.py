"""
재정렬 단계: 다중 신호 기반 최종 스코어링

이 단계에서는 게이팅을 통과한 후보들을 대상으로
의미적 유사성, 속성 일치율, 키워드 겹침을 종합하여 최종 순위를 결정합니다.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from services.query_understanding import QueryAttributes
from services.attribute_extraction import (
    calculate_attribute_match_score,
    calculate_keyword_overlap,
    extract_attributes_from_text,
)


@dataclass
class RerankingResult:
    """재정렬 결과"""
    item_id: int
    final_score: float  # 최종 점수 (0.0 ~ 1.0)
    semantic_similarity: float  # 임베딩 유사도
    attribute_match_score: float  # 속성 일치 점수
    keyword_overlap: float  # 키워드 겹침
    gating_bonus: float  # 게이팅 단계 보너스 (정규화됨)


# 재정렬 가중치
WEIGHT_SEMANTIC = 0.5  # 의미적 유사도 가중치
WEIGHT_ATTRIBUTE = 0.3  # 속성 일치 가중치
WEIGHT_KEYWORD = 0.2  # 키워드 겹침 가중치


def normalize_gating_bonus(gating_bonus: float) -> float:
    """
    게이팅 보너스 점수를 0.0 ~ 1.0 범위로 정규화
    
    게이팅 보너스는 COLOR_MATCH_BONUS (20) + PATTERN_MATCH_BONUS (15) + BRAND_MATCH_BONUS (25) = 최대 60
    """
    max_bonus = 60.0  # 모든 속성 일치 시 최대 보너스
    if max_bonus == 0:
        return 0.0
    return min(gating_bonus / max_bonus, 1.0)


def normalize_semantic_similarity(similarity: float) -> float:
    """
    FAISS 유사도를 0.0 ~ 1.0 범위로 정규화
    
    FAISS IndexFlatIP는 내적 값을 반환하며, BGE-M3 정규화 임베딩 사용 시
    대략 0.3 ~ 0.95 범위를 가집니다.
    """
    # Min-Max 정규화 (0.3 ~ 0.95 범위를 0.0 ~ 1.0으로)
    min_sim = 0.3
    max_sim = 0.95
    
    if similarity <= min_sim:
        return 0.0
    if similarity >= max_sim:
        return 1.0
    
    return (similarity - min_sim) / (max_sim - min_sim)


def rerank_items(
    candidate_item_ids: List[int],
    query_attrs: QueryAttributes,
    semantic_scores: Dict[int, float],  # {item_id: FAISS 유사도}
    item_metadata: Dict[int, Dict],  # {item_id: {'description': ..., 'item_name': ...}}
    gating_results: Dict[int, any] = None  # {item_id: GatingResult} (선택적)
) -> List[RerankingResult]:
    """
    후보 항목들을 재정렬
    
    Args:
        candidate_item_ids: 게이팅을 통과한 item_id 리스트
        query_attrs: 쿼리 속성
        semantic_scores: FAISS 임베딩 유사도 딕셔너리
        item_metadata: 항목 메타데이터
        gating_results: 게이팅 결과 (선택적, 보너스 점수 포함)
    
    Returns:
        최종 점수 순으로 정렬된 RerankingResult 리스트
    """
    results = []
    
    for item_id in candidate_item_ids:
        # 1. 의미적 유사도 (정규화)
        semantic_sim = semantic_scores.get(item_id, 0.0)
        normalized_semantic = normalize_semantic_similarity(semantic_sim)
        
        # 2. 속성 일치 점수 (구조화된 메타데이터 활용)
        if item_id not in item_metadata:
            attribute_match = 0.0
            keyword_overlap = 0.0
        else:
            metadata = item_metadata[item_id]
            item_text = f"{metadata.get('item_name', '')} {metadata.get('description', '')}".strip()
            item_attrs = extract_attributes_from_text(item_text)
            
            # 구조화된 브랜드 데이터 활용
            item_brand = metadata.get('brand')
            if item_brand and item_brand.strip():
                item_brand_normalized = item_brand.strip().lower()
                # 브랜드 키워드 사전과 매칭
                from services.query_understanding import BRAND_KEYWORDS
                for brand_key, variants in BRAND_KEYWORDS.items():
                    for variant in variants:
                        if variant.lower() == item_brand_normalized:
                            if 'brand' not in item_attrs:
                                item_attrs['brand'] = []
                            if brand_key not in item_attrs['brand']:
                                item_attrs['brand'].append(brand_key)
                            break
            
            attribute_match = calculate_attribute_match_score(
                query_attrs.attributes,
                item_attrs
            )
            
            # 3. 키워드 겹침
            keyword_overlap = calculate_keyword_overlap(
                query_attrs.keywords,
                item_text
            )
        
        # 4. 게이팅 보너스 (정규화)
        gating_bonus = 0.0
        if gating_results and item_id in gating_results:
            gating_result = gating_results[item_id]
            gating_bonus = normalize_gating_bonus(gating_result.attribute_bonus)
        
        # 5. 최종 점수 계산
        # 속성 일치 점수에 게이팅 보너스를 일부 반영 (추가 신호로 활용)
        enhanced_attribute = min(attribute_match + (gating_bonus * 0.2), 1.0)
        
        final_score = (
            WEIGHT_SEMANTIC * normalized_semantic +
            WEIGHT_ATTRIBUTE * enhanced_attribute +
            WEIGHT_KEYWORD * keyword_overlap
        )
        
        results.append(RerankingResult(
            item_id=item_id,
            final_score=final_score,
            semantic_similarity=normalized_semantic,
            attribute_match_score=attribute_match,
            keyword_overlap=keyword_overlap,
            gating_bonus=gating_bonus
        ))
    
    # 최종 점수 순으로 정렬 (내림차순)
    results.sort(key=lambda x: x.final_score, reverse=True)
    
    return results

