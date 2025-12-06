# GPU 지원 CUDA 베이스 이미지
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정 (pip 캐시 비활성화)
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# 필수 패키지 설치 및 캐시 정리 (한 번에 처리하여 레이어 최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# pip 업그레이드 및 requirements 설치 (한 번에 처리)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip /root/.cache/huggingface /tmp/* /var/tmp/* && \
    find /usr/local/lib/python3.*/dist-packages -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.*/dist-packages -name "*.pyc" -delete 2>/dev/null || true

# 소스 복사
COPY . .

# 포트 오픈
EXPOSE 5001

# Gunicorn으로 Flask 앱 실행 (멀티 워커)
# 워커 수는 환경 변수로 설정 가능 (기본값: 6)
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
