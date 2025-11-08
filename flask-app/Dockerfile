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

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 포트 오픈
EXPOSE 5001

# Flask 앱 실행
CMD ["python", "app.py"]
