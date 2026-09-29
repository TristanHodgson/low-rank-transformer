# Force the AMD64 architecture to bypass registry manifest bugs
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

# Install PyTorch and override the externally-managed environment block
RUN pip3 install --break-system-packages --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest of your requirements
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Copy repository code (excluding weights)
COPY modules/ /app/modules/
COPY api/ /app/api/

ENV WEIGHTS_DIR=/app/weights
EXPOSE 8000

CMD ["uvicorn", "api.fast-api:app", "--host", "0.0.0.0", "--port", "8000"]