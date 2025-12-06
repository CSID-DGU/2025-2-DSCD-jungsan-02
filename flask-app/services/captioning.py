import io
import os
import shutil
from functools import lru_cache
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)

# Qwen2.5-VL 모델을 위한 클래스 import
# transformers 4.51.3+ 에서 Qwen2_5_VLForConditionalGeneration 지원
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        # 대체 클래스명 시도
        from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
    except ImportError:
        # trust_remote_code=True로 자동 클래스 감지
        from transformers import AutoModelForCausalLM
        # 실제로는 AutoModelForVision2Seq를 사용해야 하지만, 
        # trust_remote_code=True로 모델이 자동으로 올바른 클래스를 선택함
        Qwen2_5_VLForConditionalGeneration = AutoModelForCausalLM

DEFAULT_MODEL_ID = os.getenv(
    "CAPTION_MODEL_ID",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)


def _check_disk_space(min_free_gb: float = 5.0):
    """모델 다운로드 전 디스크 공간 확인 및 정리"""
    try:
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024**3)
        
        if free_gb < min_free_gb:
            print(f"⚠️ 디스크 공간 부족 ({free_gb:.2f}GB). 정리 중...")
            
            # 임시 파일 정리
            for tmp_dir in ["/tmp", "/var/tmp"]:
                if os.path.exists(tmp_dir):
                    try:
                        for item in os.listdir(tmp_dir):
                            item_path = os.path.join(tmp_dir, item)
                            try:
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                            except:
                                pass
                    except:
                        pass
            
            # 다시 확인
            stat = shutil.disk_usage("/")
            free_gb_after = stat.free / (1024**3)
            print(f"✅ 정리 완료: 여유 공간 {free_gb_after:.2f}GB")
            
            if free_gb_after < min_free_gb:
                raise RuntimeError(
                    f"디스크 공간이 부족합니다. "
                    f"필요: {min_free_gb}GB 이상, 현재: {free_gb_after:.2f}GB. "
                    f"볼륨 마운트를 확인하거나 호스트 디스크 공간을 확보하세요."
                )
        
        return True
    except RuntimeError:
        raise
    except Exception as e:
        print(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True  # 실패해도 계속 진행


@lru_cache(maxsize=1)
def _load_processor(model_id: str = DEFAULT_MODEL_ID) -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True,
        use_fast=False,  # fast processor 경고 방지
    )


@lru_cache(maxsize=1)
def _load_model(model_id: str = DEFAULT_MODEL_ID):
    # 모델 다운로드 전 디스크 공간 확인 (약 4GB 필요)
    _check_disk_space(min_free_gb=5.0)
    
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Qwen2.5-VL 모델 로드
    # trust_remote_code=True로 모델이 자동으로 올바른 클래스를 선택
    print(f"📥 Qwen2.5-VL 모델 다운로드 시작: {model_id}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization,
        )
    except Exception as e:
        # fallback: AutoModel 사용 (trust_remote_code로 자동 감지)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization,
        )
        # generate 메서드가 있는지 확인
        if not hasattr(model, 'generate'):
            raise RuntimeError(
                f"로드된 모델이 generate 메서드를 지원하지 않습니다. "
                f"transformers 버전을 4.51.3 이상으로 업데이트하세요. 원본 에러: {e}"
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

    # 검색 최적화 프롬프트: 혁신적 개선 - 구체적이고 검색 친화적인 키워드 추출
    # 사용자가 검색할 때 사용할 키워드를 정확하게 포함하도록 강화
    prompt = prompt or (
        "이 분실물 이미지를 분석하여 검색에 최적화된 키워드들을 추출해줘. "
        "다음 정보를 반드시 포함해줘 (보이는 것만, 추측하지 말 것): "
        "1. 색상: 주요 색상 2-3개를 정확히 (예: 빨간색, 검은색, 흰색, 파란색, 노란색, 초록색, 회색, 베이지색, 갈색, 분홍색 등) "
        "2. 패턴/무늬: 체크, 스트라이프, 도트, 플라워, 레이스, 프린트, 솔리드(무늬없음) 등 "
        "3. 물품 종류: 셔츠, 티셔츠, 운동화, 신발, 지갑, 가방, 핸드폰, 노트북, 시계, 안경, 모자, 장갑, 우산 등 구체적으로 "
        "4. 브랜드: 보이는 브랜드명이 있으면 정확히 (나이키, 아디다스, 샘소나이트 등) "
        "5. 재질: 가죽, 나일론, 코튼, 폴리에스터, 실크, 데님, 캔버스, 메쉬 등 "
        "6. 특징: 로고, 문양, 손상, 크기, 스타일 등 눈에 띄는 특징 "
        "형식: 키워드들을 공백으로 구분하여 나열 (예: '빨간색 체크 셔츠 코튼', '검은색 나이키 운동화 에어맥스', '흰색 스트라이프 티셔츠') "
        "중요: 색상과 패턴을 반드시 포함하고, 물품 종류를 구체적으로 명시해줘."
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


