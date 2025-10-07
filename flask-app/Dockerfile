# 베이스 이미지 (Python + pip)
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 포트 오픈
EXPOSE 5001

# Flask 앱 실행
CMD ["python", "app.py"]
