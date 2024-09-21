# Use an official PyTorch base image with CUDA support (if needed)
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install system dependencies (e.g., for Ray, curl for health checks, etc.)
RUN apt-get update && \
    apt-get install -y \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ray and Ray[tune]
RUN pip install "ray[default]" "ray[tune]" "ray[train]" torch torchvision

# Install additional ML tools if necessary
RUN pip install \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    tensorboard

# Expose ports for Ray Dashboard and TensorBoard
EXPOSE 8265 6006

# Copy the application code
COPY . /app

# Start the Ray cluster
ENTRYPOINT ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0"]

# Run the training script as the main command
CMD ["python", "src/training/local_training.py"]
