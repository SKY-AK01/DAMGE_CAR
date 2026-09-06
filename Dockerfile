FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install necessary system packages for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 wheels FIRST (must match base image CUDA 11.8)
# Default pip install of ultralytics pulls cu12/cu13 torch which won't see the GPU!
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file and install remaining dependencies (no torch — already installed)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt azureml-core azureml-dataprep


# YOLOv11 and PyTorch are in requirements.txt, but let's ensure torchvision is explicitly available.
# We also need azure-identity and azure-ai-ml for SDK v2 scripts, 
# but usually, these run on the control node. The training environment only needs what the script uses.
