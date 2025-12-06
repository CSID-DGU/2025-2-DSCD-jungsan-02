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

# requirements 설치 (디스크 공간 절약을 위해 빌드 중 정리)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /tmp/* /var/tmp/* && \
    find /usr/local/lib/python3.*/dist-packages -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.*/dist-packages -name "*.pyc" -delete 2>/dev/null || true

# 소스 복사
COPY . .

# 포트 오픈
EXPOSE 5001

# Gunicorn으로 Flask 앱 실행 (멀티 워커)
# 워커 수는 환경 변수로 설정 가능 (기본값: 6)
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
