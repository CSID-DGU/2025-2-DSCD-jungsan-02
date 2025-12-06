import io
import os
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

DEFAULT_MODEL_ID = os.getenv(
    "CAPTION_MODEL_ID",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)


@lru_cache(maxsize=1)
def _load_processor(model_id: str = DEFAULT_MODEL_ID) -> AutoProcessor:
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_model(model_id: str = DEFAULT_MODEL_ID) -> AutoModelForCausalLM:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization,
    )
    model.eval()
    return model


def generate_caption(
    image_bytes: bytes,
    prompt: Optional[str] = None,
) -> str:
    """
    Generate a Korean caption for the given image.
    검색 최적화된 구조화된 프롬프트로 개선하여 검색 성능 향상.
    리소스 사용량은 동일하지만 검색 정확도가 크게 향상됨.
    """
    if not image_bytes:
        raise ValueError("이미지 데이터가 비어 있습니다.")

    # 검색 최적화 프롬프트: 키워드 중심, 일반적이고 유연한 형식
    # 오버피팅 방지: 구체적인 예시를 줄이고 일반적인 지시로 변경
    # 사용자가 검색할 때 사용할 키워드를 우선적으로 포함하도록 지시
    prompt = prompt or (
        "이 분실물 이미지를 분석하여 검색에 유용한 키워드들을 추출해줘. "
        "다음 정보를 우선순위대로 포함해줘: "
        "- 주요 색상 (2-3개) "
        "- 브랜드명 (보이는 경우에만) "
        "- 물품 종류/카테고리 "
        "- 재질/소재 (가죽, 플라스틱, 금속 등) "
        "- 눈에 띄는 특징 (로고, 문양, 손상, 크기 등) "
        "키워드들을 공백으로 구분하여 나열해줘. 문장 형태가 아닌 키워드 나열 형식으로 작성해줘. "
        "이미지에서 확실히 보이는 정보만 포함하고, 추측하지 말아줘."
    )

    model_id = DEFAULT_MODEL_ID
    processor = _load_processor(model_id)
    model = _load_model(model_id)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_length:]

    outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    caption = outputs[0].strip()
    return caption


