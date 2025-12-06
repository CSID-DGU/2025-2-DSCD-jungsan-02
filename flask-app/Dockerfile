# GPU 지원 CUDA 베이스 이미지
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

# python 명령어 심볼릭 링크 (python3 -> python)
RUN ln -s /usr/bin/python3 /usr/bin/python

# requirements 설치 (디스크 공간 절약을 위해 단계별 설치 및 정리)
COPY requirements.txt .

# 1단계: 기본 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir Flask==2.3.2 gunicorn==21.2.0 flask-cors==4.0.0 && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 2단계: NumPy 및 기본 과학 계산 라이브러리
RUN pip install --no-cache-dir "numpy>=1.24.3,<2.0.0" Pillow>=10.0.0 scipy scikit-learn && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 3단계: FAISS 설치
RUN pip install --no-cache-dir faiss-cpu==1.7.4 && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 4단계: Torch 및 관련 패키지 (가장 큰 패키지)
RUN pip install --no-cache-dir "torch>=2.6.0" "torchvision>=0.21.0" && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 5단계: Transformers 및 관련 패키지
RUN pip install --no-cache-dir "transformers>=4.46.0" "accelerate>=0.30.1" "huggingface-hub>=0.24.6" "safetensors>=0.4.3" && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 6단계: Sentence Transformers 및 기타 패키지
RUN pip install --no-cache-dir "sentence-transformers>=2.7.0" "bitsandbytes>=0.43.1" && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 7단계: 나머지 패키지
RUN pip install --no-cache-dir soynlp==0.0.493 "qwen-vl-utils>=0.0.10" sentencepiece==0.1.99 && \
    pip cache purge && rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# 설치 확인
RUN python -c "import sentence_transformers; print('sentence_transformers:', sentence_transformers.__version__)" && \
    python -c "import torch; print('torch:', torch.__version__)" && \
    python -c "import transformers; print('transformers:', transformers.__version__)"

# 소스 복사
COPY . .

# 포트 오픈
EXPOSE 5001

# Gunicorn으로 Flask 앱 실행 (멀티 워커)
# 워커 수는 환경 변수로 설정 가능 (기본값: 6)
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
