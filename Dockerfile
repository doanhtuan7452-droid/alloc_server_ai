# Stage 1: Build wheel files and dependencies
FROM python:3.14-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final minimal runtime environment
FROM python:3.14-slim as runner

WORKDIR /app

# Safely copy compiled user packages from builder
COPY --from=builder /root/.local /root/.local

# Add local bin to Path and prevent python buffering
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project files
COPY app/ app/
COPY model_ai/ model_ai/
COPY api_server.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=\"$FORWARDED_ALLOW_IPS\""]
