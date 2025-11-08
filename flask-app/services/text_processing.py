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


def preprocess_text(raw_text: Optional[str]) -> str:
    """Apply typo correction, spacing, and emoji normalization."""
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text:
        return ""

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

    # Remove unsupported characters
    corrected = re.sub(r"[^가-힣0-9a-zA-Zㄱ-ㅎㅏ-ㅣ .,!?~]", "", corrected)

    # Fix basic spacing & repeated emoticons
    corrected = corrected.replace(" ", "")
    corrected = re.sub(r'([가-힣0-9a-zA-Z])([,.!?])', r'\1 \2', corrected)
    corrected = re.sub(r'([,.!?])([가-힣0-9a-zA-Z])', r'\1 \2', corrected)
    corrected = emoticon_normalize(corrected, num_repeats=1)

    return corrected.strip()


