# Gunicorn 설정 파일
import multiprocessing
import os

# 워커 수 설정
# GPU 사용 시 메모리 제한으로 인해 워커 수를 제한해야 함
# 
# 메모리 계산 (4-bit quantization 기준):
# - Qwen2.5-VL-1.8B (4-bit): ~1-1.5GB
# - BGE-M3: ~1-1.5GB
# - 각 워커당 총: ~2-3GB
# 
# 15GB VRAM 기준:
# - 안전: 3-4개 워커 (각 3-4GB, 시스템 오버헤드 고려)
# - 실제로는 각 워커가 독립적으로 모델을 로드하므로 더 많은 메모리 필요
# - 시스템 메모리(RAM)도 고려해야 함
workers = int(os.getenv('GUNICORN_WORKERS', 3))  # 기본값: 3개 워커 (안정성 우선)

# 워커 타입: sync (동기) 또는 gevent (비동기)
worker_class = 'sync'

# 각 워커당 스레드 수 (sync 워커의 경우)
threads = 1

# 타임아웃 설정 (초)
timeout = 300  # 5분 (이미지 처리 시간 고려)

# 바인드 주소
bind = '0.0.0.0:5001'

# 로그 설정
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# 프로세스 이름
proc_name = 'flask-app'

# 워커 재시작 설정
max_requests = 1000  # 1000개 요청 후 워커 재시작 (메모리 누수 방지)
max_requests_jitter = 100

# 백로그 (대기 큐 크기)
backlog = 2048

