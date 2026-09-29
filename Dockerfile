# Force AMD64 architecture and use the stable Ubuntu 22.04 base
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (Ubuntu 22.04 natively permits pip installs here)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest of your requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy repository code (excluding weights)
COPY modules/ /app/modules/
COPY api/ /app/api/

ENV WEIGHTS_DIR=/app/weights
EXPOSE 8000

CMD ["uvicorn", "api.fast-api:app", "--host", "0.0.0.0", "--port", "8000"]