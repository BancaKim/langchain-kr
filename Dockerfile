# 베이스 이미지로 Python 사용
FROM python:3.11.9

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -

# PATH에 Poetry 추가
ENV PATH="${PATH}:/root/.local/bin"

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 관련 파일 복사
COPY pyproject.toml poetry.lock ./

# Poetry 설정: 가상 환경을 프로젝트 디렉토리 내에 생성
RUN poetry config virtualenvs.create false

# 프로젝트 의존성 설치
RUN poetry install --only main --no-interaction --no-ansi

# 프로젝트 파일 복사
COPY . .

# 애플리케이션 실행을 위한 명령어
CMD ["poetry", "run", "uvicorn", "controllers.main:app", "--host", "0.0.0.0", "--port", "8000"]