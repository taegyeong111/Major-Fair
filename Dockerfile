# 1. 베이스 이미지 설정
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 및 디렉토리 통째로 복사
COPY . .

# 5. 컨테이너 실행 시 가상 데이터셋 자동 생성 및 Flask 실행
EXPOSE 5000
CMD ["python", "app.py"]