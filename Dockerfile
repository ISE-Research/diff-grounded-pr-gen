FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/ISE-Research/diff-grounded-pr-gen"
LABEL org.opencontainers.image.description="ICSME 2026 replication package for diff-grounded pull-request description generation"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /artifact

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
