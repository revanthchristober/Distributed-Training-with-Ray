# **Distributed Model Training and Evaluation**

This repository contains a comprehensive setup for distributed training of machine learning models using **Ray.io**. It includes scripts for local and distributed training, model evaluation, and deployment using **Docker**. The project supports distributed hyperparameter tuning, profiling, and visualizations of model architectures and performance metrics.

## **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Setup](#setup)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Running Local Training](#running-local-training)
  - [Running Distributed Training](#running-distributed-training)
  - [Model Evaluation](#model-evaluation)
  - [Profiling](#profiling)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Deployment](#deployment)
  - [Ray Cluster Setup](#ray-cluster-setup)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [Contributing](#contributing)
- [License](#license)

---

## **Overview**

This project enables scalable training and evaluation of machine learning models with the following key features:

1. **Distributed Training** using Ray.io across multiple nodes.
2. **Custom Model Architectures** including RNNs, CNNs, and transformers.
3. **Evaluation Metrics** such as precision, recall, accuracy, and F1-score.
4. **Dockerized Environment** for easy setup and deployment on cloud or local clusters.
5. **Profiling Tools** for monitoring resource usage and performance during training.

---

## **Features**

- **Distributed Training**: Run training across multiple CPUs or GPUs using Ray.io and distributed data loaders.
- **Model Evaluation**: Includes scripts for evaluating model performance using accuracy, F1-score, and confusion matrices.
- **Visualization Tools**: Generate plots to visualize training progress, loss curves, and model architecture.
- **Hyperparameter Tuning**: Utilize Ray Tune to find the best hyperparameters through distributed tuning.
- **Profiling and Monitoring**: Integrated profiling utilities for tracking memory, CPU, and GPU usage.
- **Deployment with Docker**: Dockerized setup for seamless deployment in any environment.

---

## **System Requirements**

- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **Ray.io**: 2.0+
- **Docker**: 20.10.0+
- **Docker Compose**: 1.29.0+
- **Pytorch**: 1.10.0+

---

## **Setup**

### **Local Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/distributed-model-training.git
   cd distributed-model-training
   ```

2. **Install Python dependencies**:
   It is recommended to use a virtual environment (e.g., `venv` or `conda`):
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ray**:
   Install Ray for distributed training:
   ```bash
   pip install ray[default]
   ```

4. **Set up environment variables** (Optional but recommended):
   - You can create a `.env` file to store environment-specific configurations such as paths, model save directories, and cloud setup information.

### **Docker Setup**

1. **Install Docker and Docker Compose**:
   Ensure that Docker and Docker Compose are installed on your system:
   ```bash
   sudo apt install docker docker-compose
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t distributed-training:latest .
   ```

3. **Run the container**:
   ```bash
   docker-compose up -d
   ```

   This will start the container with all required services, including Ray.io head and worker nodes.

---

## **Usage**

### **Running Local Training**

To run the local training script, use:
```bash
bash scripts/run_local_training.sh
```

This script will:
- Load the dataset.
- Initialize the model.
- Start the training loop on the local machine.

### **Running Distributed Training**

To run distributed training using Ray.io, use:
```bash
bash scripts/run_distributed_training.sh
```

This will:
- Set up Ray on the specified nodes.
- Distribute the training workload across the cluster.
- Train the model in parallel across multiple CPUs/GPUs.

### **Model Evaluation**

To evaluate the model after training, run:
```bash
bash scripts/evaluate_model.sh
```

This will:
- Load the trained model checkpoint.
- Run evaluation on the test dataset.
- Output metrics such as accuracy, precision, recall, and F1-score.

### **Profiling**

To profile the model during training:
```bash
python src/utils/profiler_utils.py
```

This will collect memory, CPU, and GPU statistics during training and save them for analysis.

### **Hyperparameter Tuning**

To run hyperparameter tuning with Ray Tune:
```bash
python src/training/distributed_training.py --tune
```

This will perform a distributed search for the best hyperparameters and output the best model configuration.

---

## **Deployment**

### **Ray Cluster Setup**

To deploy the Ray.io cluster for distributed training, run the following:
```bash
bash scripts/ray_cluster_start.sh
```

This script will:
- Start Ray head and worker nodes.
- Initialize the cluster configuration.
- Prepare the environment for distributed training.

### **Docker Compose Deployment**

For cloud or local multi-container deployment using Docker Compose:
```bash
docker-compose up -d
```

This will launch the Ray head node and multiple worker nodes in separate containers for distributed training.

---

## **Contributing**

Contributions are welcome! If you would like to contribute, please fork the repository and create a pull request. Make sure your code adheres to the following guidelines:

1. Write clear, concise commit messages.
2. Include unit tests for any new functionality.
3. Ensure that all tests pass before submitting your pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
