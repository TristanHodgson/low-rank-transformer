FROM --platform=linux/amd64 nvidia/cuda:13.4.1-runtime-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install torch torchvision

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy repository modules and API script
COPY modules/ /app/modules/
COPY api/ /app/api/

ENV WEIGHTS_DIR=/app/weights
EXPOSE 8000

CMD ["uvicorn", "api.fast-api:app", "--host", "0.0.0.0", "--port", "8000"]