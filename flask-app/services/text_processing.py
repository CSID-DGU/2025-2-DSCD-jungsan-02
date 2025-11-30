import re
from functools import lru_cache
from typing import Optional

import torch
from soynlp.normalizer import emoticon_normalize
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "j5ng/et5-typos-corrector"


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


def preprocess_text(raw_text: Optional[str], use_typo_correction: bool = True) -> str:
    """
    텍스트 전처리 (맞춤법 교정, 공백 정규화)
    
    Args:
        raw_text: 전처리할 텍스트
        use_typo_correction: 맞춤법 교정 사용 여부 (기본값: True)
                           False로 설정하면 공백 정규화만 수행
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text:
        return ""

    corrected = text

    # 맞춤법 교정 (선택적)
    if use_typo_correction:
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
                    num_beams=5,
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
    # 공백을 완전히 제거하지 않고, 적절한 공백만 유지
    corrected = re.sub(r'([가-힣0-9a-zA-Z])([,.!?])', r'\1 \2', corrected)
    corrected = re.sub(r'([,.!?])([가-힣0-9a-zA-Z])', r'\1 \2', corrected)
    corrected = emoticon_normalize(corrected, num_repeats=1)

    return corrected.strip()


