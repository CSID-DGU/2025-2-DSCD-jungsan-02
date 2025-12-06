import re
from functools import lru_cache
from typing import Optional, List

import torch
from soynlp.normalizer import emoticon_normalize
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "j5ng/et5-typos-corrector"

# 검색 최적화: 불용어 리스트 (검색에 도움이 안 되는 단어)
# 조사, 어미, 분실 관련 동사 등 검색에 불필요한 단어 제거
STOPWORDS = {
    # 조사
    '에서', '을', '를', '이', '가', '의', '에', '와', '과', '로', '으로',
    '은', '는', '도', '만', '까지', '부터', '에게', '께', '한테', '처럼', '같이',
    # 어미/종결어미
    '입니다', '이에요', '예요', '이야', '야', '다', '어', '아', '네', '지',
    # 분실 관련 동사 (검색 키워드가 아님)
    '잃어버린', '잃어버린', '분실한', '잃은', '찾는', '찾고있는', '찾고', '찾아',
    '발견한', '발견', '습득한', '습득', '주운', '주웠', '찾아주세요', '찾아요',
    # 일반 동사/형용사 (검색에 불필요)
    '있어요', '있습니다', '있어', '없어요', '없습니다', '없어',
    '같아요', '같습니다', '같아', '다른', '다르',
    # 기타 불필요한 단어
    '제', '내', '저의', '나의', '이것', '저것', '그것', '이거', '저거', '그거',
    '여기', '저기', '거기', '어디', '언제', '어떤', '어떻게',
}

# 검색 키워드 확장 사전 (동의어/유사어)
# 분실물 검색에 자주 사용되는 키워드들의 동의어 확장
KEYWORD_EXPANSION = {
    # 신발류
    '운동화': ['운동화', '신발', '스니커즈', '스니커', '운동화', '구두', '신발'],
    '구두': ['구두', '신발', '운동화', '하이힐', '로퍼'],
    '부츠': ['부츠', '장화', '부티', '워커'],
    '샌들': ['샌들', '슬리퍼', '플립플롭'],
    
    # 지갑류
    '지갑': ['지갑', '반지갑', '장지갑', '카드지갑', '명함지갑', '코인지갑'],
    '반지갑': ['반지갑', '지갑', '카드지갑'],
    '장지갑': ['장지갑', '지갑', '명함지갑'],
    
    # 가방류
    '가방': ['가방', '백팩', '크로스백', '토트백', '숄더백', '백', '서류가방'],
    '백팩': ['백팩', '배낭', '가방', '책가방'],
    '크로스백': ['크로스백', '가방', '메신저백'],
    '토트백': ['토트백', '가방', '쇼핑백'],
    '핸드백': ['핸드백', '가방', '클러치백'],
    
    # 전자기기
    '핸드폰': ['핸드폰', '스마트폰', '휴대폰', '아이폰', '갤럭시', '폰', '전화기'],
    '스마트폰': ['스마트폰', '핸드폰', '휴대폰', '폰'],
    '아이폰': ['아이폰', '스마트폰', '핸드폰', '폰'],
    '갤럭시': ['갤럭시', '스마트폰', '핸드폰', '폰'],
    '노트북': ['노트북', '랩톱', '컴퓨터', '맥북', '그램'],
    '태블릿': ['태블릿', '아이패드', '갤럭시탭', '패드'],
    '이어폰': ['이어폰', '헤드폰', '에어팟', '버즈', '이어버드'],
    '충전기': ['충전기', '케이블', '어댑터', '전원선'],
    
    # 액세서리
    '시계': ['시계', '손목시계', '스마트워치', '워치'],
    '안경': ['안경', '선글라스', '렌즈', '돋보기'],
    '목걸이': ['목걸이', '펜던트', '체인'],
    '반지': ['반지', '링'],
    '귀걸이': ['귀걸이', '이어링'],
    
    # 기타 일상용품
    '열쇠': ['열쇠', '키', '키링'],
    '우산': ['우산', '양산', '장우산', '접이식우산'],
    '지갑': ['지갑', '반지갑', '장지갑'],
    '마스크': ['마스크', '마스크', 'KF94', 'KF80'],
    '장갑': ['장갑', '글러브', '손목장갑'],
    '모자': ['모자', '캡', '야구모자', '비니'],
    '스카프': ['스카프', '목도리', '머플러'],
    
    # 의류
    '옷': ['옷', '의류', '상의', '하의'],
    '재킷': ['재킷', '자켓', '아우터'],
    '코트': ['코트', '아우터', '외투'],
    '후드': ['후드', '후드티', '후드집업'],
    
    # 색상 (일부 주요 색상)
    '검은색': ['검은색', '검정', '검정색', '블랙'],
    '흰색': ['흰색', '하양', '하얀색', '화이트'],
    '빨간색': ['빨간색', '빨강', '빨강색', '레드'],
    '파란색': ['파란색', '파랑', '파랑색', '블루'],
    '노란색': ['노란색', '노랑', '노랑색', '옐로우'],
    '초록색': ['초록색', '초록', '녹색', '그린'],
    
    # 브랜드 (주요 브랜드)
    '나이키': ['나이키', '니케', 'nike'],
    '아디다스': ['아디다스', '아디', 'adidas'],
    '샘소나이트': ['샘소나이트', 'samsonite'],
    '구찌': ['구찌', 'gucci'],
    '프라다': ['프라다', 'prada'],
    '루이비통': ['루이비통', 'lv', 'louis vuitton'],
}


@lru_cache(maxsize=1)
def _load_tokenizer() -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def _load_model() -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def preprocess_text(
    raw_text: Optional[str], 
    use_typo_correction: bool = True,
    optimize_for_search: bool = True
) -> str:
    """
    텍스트 전처리 (맞춤법 교정, 공백 정규화, 검색 최적화)
    
    리소스 절약: 짧은 텍스트나 이미 잘 정제된 텍스트는 맞춤법 교정 스킵
    
    Args:
        raw_text: 전처리할 텍스트
        use_typo_correction: 맞춤법 교정 사용 여부 (기본값: True)
                           짧은 텍스트(10자 이하)는 자동으로 스킵하여 리소스 절약
        optimize_for_search: 검색 최적화 적용 여부 (기본값: True)
                            불용어 제거, 키워드 정규화 등
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text:
        return ""

    corrected = text

    # 맞춤법 교정 (리소스 절약: 짧은 텍스트나 키워드 형태는 스킵)
    if use_typo_correction:
        # 짧은 텍스트(10자 이하)나 키워드 형태는 맞춤법 교정 스킵하여 리소스 절약
        should_correct = (
            len(text) > 10 and  # 긴 텍스트만 교정
            not text.isdigit() and  # 숫자만 있는 경우 스킵
            ' ' in text  # 공백이 있는 문장 형태만 교정
        )
        
        if should_correct:
            try:
                tokenizer = _load_tokenizer()
                model = _load_model()
                device = model.device

                # Typo correction via T5 model
                encoding = tokenizer(
                    f"맞춤법을 고쳐주세요: {text}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                input_ids = encoding.input_ids.to(device)
                attention_mask = encoding.attention_mask.to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=128,
                        num_beams=3,  # 5 -> 3으로 줄여서 리소스 절약
                        early_stopping=True,
                    )

                corrected = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as e:
                # 맞춤법 교정 실패 시 원본 사용
                print(f"⚠️ 맞춤법 교정 실패, 원본 사용: {e}")
                corrected = text

    # Remove unsupported characters (공백은 유지)
    corrected = re.sub(r"[^가-힣0-9a-zA-Zㄱ-ㅎㅏ-ㅣ .,!?~]", "", corrected)

    # 공백 정규화 (연속된 공백을 하나로, 하지만 공백 자체는 유지)
    corrected = re.sub(r'\s+', ' ', corrected)
    
    # Fix basic spacing & repeated emoticons
    corrected = re.sub(r'([가-힣0-9a-zA-Z])([,.!?])', r'\1 \2', corrected)
    corrected = re.sub(r'([,.!?])([가-힣0-9a-zA-Z])', r'\1 \2', corrected)
    corrected = emoticon_normalize(corrected, num_repeats=1)

    # 검색 최적화: 불용어 제거 및 키워드 정규화
    if optimize_for_search:
        words = corrected.split()
        # 불용어 제거 (검색에 도움이 안 되는 단어)
        filtered_words = [w for w in words if w not in STOPWORDS]
        corrected = ' '.join(filtered_words)

    return corrected.strip()


def expand_search_query(query: str) -> List[str]:
    """
    검색 쿼리 확장 (동의어/유사어 추가)
    
    리소스 절약: 사전 기반 확장으로 LLM 없이 빠르게 처리
    검색 성능 향상: 동의어를 포함하여 검색 범위 확대
    
    Args:
        query: 원본 검색 쿼리
        
    Returns:
        확장된 검색 쿼리 리스트 (원본 포함)
    """
    if not query:
        return []
    
    expanded_queries = [query]  # 원본 쿼리 포함
    
    # 키워드 확장 사전을 사용하여 동의어 추가
    words = query.split()
    added_queries = set()  # 중복 방지용
    
    for word in words:
        word_lower = word.lower()  # 대소문자 구분 없이 검색
        
        # 정확히 일치하는 경우
        if word in KEYWORD_EXPANSION:
            for synonym in KEYWORD_EXPANSION[word]:
                if synonym != word:  # 중복 제거
                    expanded_query = query.replace(word, synonym)
                    if expanded_query not in added_queries and expanded_query != query:
                        expanded_queries.append(expanded_query)
                        added_queries.add(expanded_query)
        
        # 소문자로 변환하여 검색 (대소문자 구분 없이)
        elif word_lower in KEYWORD_EXPANSION:
            for synonym in KEYWORD_EXPANSION[word_lower]:
                if synonym.lower() != word_lower:
                    expanded_query = query.replace(word, synonym)
                    if expanded_query not in added_queries and expanded_query != query:
                        expanded_queries.append(expanded_query)
                        added_queries.add(expanded_query)
    
    # 최대 8개로 제한하여 리소스 절약 (원본 포함)
    return expanded_queries[:8]


