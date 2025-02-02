FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV CUDA_HOME=/usr/local/cuda-11.3/

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    python3.8 \
    python3-pip \
    python3-venv \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone GroundingDINO repo
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install networkx==2.6.3

# Install GroundingDINO requirements
RUN cd GroundingDINO && \
    pip install -r requirements.txt && \
    pip install -e .

# Download pre-trained weights
RUN mkdir -p /app/weights && \
    wget -P /app/weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Environment variables
ENV PYTHONPATH=/app/GroundingDINO
ENV CUDA_VISIBLE_DEVICES=0

# Use JSON array format for CMD
CMD ["bash", "-c", "cd GroundingDINO && python3 demo/gradio_app.py"]
