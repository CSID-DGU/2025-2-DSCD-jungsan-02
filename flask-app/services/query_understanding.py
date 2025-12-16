"""
쿼리 이해 단계: 사용자 검색어에서 구조화된 속성을 명시적으로 추출

이 모듈은 검색 쿼리를 분석하여 다음 정보를 추출합니다:
- 카테고리 (category)
- 속성 (color, pattern, brand, material 등)
- 키워드
- 컨텍스트 (장소, 상황 등)
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class QueryAttributes:
    """쿼리에서 추출된 구조화된 속성"""
    category: Optional[str] = None  # ItemCategory enum 값 (WALLET, PHONE, BAG 등)
    attributes: Dict[str, List[str]] = field(default_factory=dict)  # color, pattern, brand, material 등
    keywords: List[str] = field(default_factory=list)  # 검색 키워드
    context: List[str] = field(default_factory=list)  # 장소/상황 정보


# 카테고리 키워드 사전 (한국어 → ItemCategory enum)
CATEGORY_KEYWORDS = {
    "WALLET": ["지갑", "반지갑", "장지갑", "카드지갑", "명함지갑", "코인지갑"],
    "PHONE": ["핸드폰", "스마트폰", "휴대폰", "아이폰", "갤럭시", "폰", "전화기", "휴대전화"],
    "CARD": ["카드", "신용카드", "체크카드", "교통카드"],
    "BAG": ["가방", "백팩", "배낭", "크로스백", "토트백", "숄더백", "서류가방", "핸드백", "클러치백"],
    "CLOTHING": ["옷", "의류", "셔츠", "티셔츠", "티", "바지", "청바지", "재킷", "코트", "후드", "상의", "하의"],
    "ETC": []  # 기타는 명시적 키워드 없음
}

# 색상 키워드 사전 (한국어 변형 포함)
COLOR_KEYWORDS = {
    "검은색": ["검은색", "검정", "검정색", "블랙", "검"],
    "흰색": ["흰색", "하양", "하얀색", "화이트", "흰"],
    "빨간색": ["빨간색", "빨강", "빨강색", "레드", "빨"],
    "파란색": ["파란색", "파랑", "파랑색", "블루", "파"],
    "노란색": ["노란색", "노랑", "노랑색", "옐로우", "노"],
    "초록색": ["초록색", "초록", "녹색", "그린", "초"],
    "회색": ["회색", "그레이", "회"],
    "베이지색": ["베이지색", "베이지", "베이지색"],
    "갈색": ["갈색", "브라운", "갈"],
    "분홍색": ["분홍색", "핑크", "분홍"],
    "주황색": ["주황색", "오렌지", "주황"],
    "보라색": ["보라색", "퍼플", "보라"],
}

# 패턴 키워드 사전
PATTERN_KEYWORDS = {
    "체크": ["체크", "체크무늬", "체크패턴", "체크무늬"],
    "스트라이프": ["스트라이프", "줄무늬", "줄", "세로줄", "가로줄", "줄무늬"],
    "도트": ["도트", "점무늬", "점"],
    "플라워": ["플라워", "꽃무늬", "꽃"],
    "레인보우": ["레인보우", "무지개"],
    "솔리드": ["솔리드", "무늬없음", "단색"],
}

# 브랜드 키워드 사전
BRAND_KEYWORDS = {
    "나이키": ["나이키", "니케", "nike", "NIKE"],
    "아디다스": ["아디다스", "아디", "adidas", "ADIDAS"],
    "샘소나이트": ["샘소나이트", "samsonite", "SAMSONITE"],
    "구찌": ["구찌", "gucci", "GUCCI"],
    "프라다": ["프라다", "prada", "PRADA"],
    "루이비통": ["루이비통", "lv", "louis vuitton", "LV"],
    "아이폰": ["아이폰", "iphone", "iPhone", "애플", "apple"],
    "갤럭시": ["갤럭시", "galaxy", "삼성", "samsung"],
}

# 장소/상황 키워드 (컨텍스트)
CONTEXT_KEYWORDS = [
    "지하철", "지하철역", "역", "기차", "버스", "택시",
    "강남역", "홍대", "명동", "신촌",
    "카페", "식당", "학교", "회사", "도서관",
    "잃어버린", "잃은", "분실한", "찾는", "찾고있는",
]


def extract_category(query: str) -> Optional[str]:
    """
    쿼리에서 카테고리를 추출
    
    Returns:
        ItemCategory enum 값 (WALLET, PHONE, BAG 등) 또는 None
    """
    query_lower = query.lower()
    
    # 각 카테고리별로 키워드 매칭
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query or keyword.lower() in query_lower:
                return category
    
    return None


def extract_colors(query: str) -> List[str]:
    """
    쿼리에서 색상을 추출
    
    Returns:
        정규화된 색상 리스트 (예: ["검은색", "빨간색"])
    """
    found_colors = []
    query_lower = query.lower()
    
    for normalized_color, variants in COLOR_KEYWORDS.items():
        for variant in variants:
            if variant in query or variant.lower() in query_lower:
                if normalized_color not in found_colors:
                    found_colors.append(normalized_color)
                break
    
    return found_colors


def extract_patterns(query: str) -> List[str]:
    """
    쿼리에서 패턴을 추출
    
    Returns:
        정규화된 패턴 리스트 (예: ["체크", "스트라이프"])
    """
    found_patterns = []
    query_lower = query.lower()
    
    for normalized_pattern, variants in PATTERN_KEYWORDS.items():
        for variant in variants:
            if variant in query or variant.lower() in query_lower:
                if normalized_pattern not in found_patterns:
                    found_patterns.append(normalized_pattern)
                break
    
    return found_patterns


def extract_brands(query: str) -> List[str]:
    """
    쿼리에서 브랜드를 추출
    
    Returns:
        정규화된 브랜드 리스트 (예: ["나이키"])
    """
    found_brands = []
    query_lower = query.lower()
    
    for normalized_brand, variants in BRAND_KEYWORDS.items():
        for variant in variants:
            # 단어 단위로 매칭 (부분 문자열 매칭 방지)
            pattern = r'\b' + re.escape(variant) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                if normalized_brand not in found_brands:
                    found_brands.append(normalized_brand)
                break
    
    return found_brands


def extract_context(query: str) -> List[str]:
    """
    쿼리에서 컨텍스트(장소, 상황)를 추출
    
    Returns:
        컨텍스트 키워드 리스트
    """
    found_context = []
    query_lower = query.lower()
    
    for context_keyword in CONTEXT_KEYWORDS:
        if context_keyword in query or context_keyword.lower() in query_lower:
            found_context.append(context_keyword)
    
    return found_context


def extract_keywords(query: str) -> List[str]:
    """
    쿼리에서 주요 키워드를 추출 (검색에 사용할 단어들)
    
    Returns:
        키워드 리스트
    """
    # 공백으로 분리
    words = query.split()
    
    # 불용어 제거 (context 키워드는 제외)
    stopwords = {
        '에서', '을', '를', '이', '가', '의', '에', '와', '과', '로', '으로',
        '은', '는', '도', '만', '까지', '부터', '에게', '께', '한테',
        '잃어버린', '잃은', '분실한', '찾는', '찾고있는', '찾고', '찾아',
        '발견한', '발견', '습득한', '습득', '주운', '주웠',
    }
    
    keywords = []
    for word in words:
        word = word.strip()
        if word and word not in stopwords:
            keywords.append(word)
    
    return keywords


def understand_query(query: str) -> QueryAttributes:
    """
    쿼리 이해: 자연어 검색어에서 구조화된 속성을 추출
    
    중요: 카테고리는 추출하지 않음 (임베딩이 이미 의미를 포착)
    
    Args:
        query: 사용자 검색어 (예: "지하철에서 잃어버린 검은색 체크 지갑")
    
    Returns:
        QueryAttributes 객체 (구조화된 속성 정보)
        - category: 항상 None (임베딩이 의미를 포착하므로 추출 불필요)
        - attributes: 색상, 패턴, 브랜드만 추출 (명시적 속성)
    
    Example:
        >>> attrs = understand_query("지하철에서 잃어버린 검은색 체크 지갑")
        >>> print(attrs.category)
        None
        >>> print(attrs.attributes['color'])
        ['검은색']
        >>> print(attrs.attributes['pattern'])
        ['체크']
    """
    if not query or not query.strip():
        return QueryAttributes()
    
    query = query.strip()
    
    # 카테고리 추출 제거 - 임베딩이 이미 의미를 포착하므로 불필요
    # category = extract_category(query)  # 제거됨
    
    # 명시적 속성만 추출 (색상, 패턴, 브랜드)
    colors = extract_colors(query)
    patterns = extract_patterns(query)
    brands = extract_brands(query)
    context = extract_context(query)
    keywords = extract_keywords(query)
    
    # 속성 딕셔너리 구성
    attributes = {}
    if colors:
        attributes['color'] = colors
    if patterns:
        attributes['pattern'] = patterns
    if brands:
        attributes['brand'] = brands
    
    return QueryAttributes(
        category=None,  # 항상 None - 카테고리는 임베딩으로 처리
        attributes=attributes,
        keywords=keywords,
        context=context
    )


def attributes_to_dict(attrs: QueryAttributes) -> Dict:
    """
    QueryAttributes를 딕셔너리로 변환 (JSON 직렬화용)
    """
    return {
        'category': attrs.category,
        'attributes': attrs.attributes,
        'keywords': attrs.keywords,
        'context': attrs.context
    }

