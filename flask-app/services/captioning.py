import io
import os
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

DEFAULT_MODEL_ID = os.getenv(
    "CAPTION_MODEL_ID",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)


@lru_cache(maxsize=1)
def _load_processor(model_id: str = DEFAULT_MODEL_ID) -> AutoProcessor:
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_model(model_id: str = DEFAULT_MODEL_ID) -> Qwen2_5_VLForConditionalGeneration:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    """
    if not image_bytes:
        raise ValueError("이미지 데이터가 비어 있습니다.")

    prompt = prompt or "이 이미지를 한국어로 한 문장으로 설명해줘."

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

    outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    caption = outputs[0].strip()
    return caption


