# **Distributed Training Guide**

## **1. Overview**
This guide provides step-by-step instructions for setting up and running distributed training using **Ray.io**. Ray enables scalable and efficient distributed training by parallelizing workloads across multiple nodes. This is particularly useful when training large models or datasets that would otherwise require substantial time and resources if run on a single machine.

### **Technologies Used**
- **Ray.io**: Distributed computing framework for scaling Python applications.
- **Torch**: PyTorch for deep learning.
- **Docker**: Containerization for reproducibility.
- **AWS/VM Instances**: Cloud infrastructure for scaling distributed training.

---

## **2. Prerequisites**

Before starting, ensure you have the following:

### **2.1 Infrastructure**
- **Multiple GPUs** or **Distributed CPU nodes** (either on-premise or cloud, such as AWS EC2 instances).
- **Ray cluster** set up across multiple nodes.

### **2.2 Software and Libraries**
- **Ray.io** (latest version)
- **PyTorch** (latest version compatible with Ray)
- **Docker** for containerized environments (recommended for cloud setups)
- **Python 3.8+**

### **2.3 File Structure**
Ensure your project has the following structure for distributed training:

```plaintext
project_root/
│
├── config/
│   ├── environment.yaml               # Configuration for distributed environment
│   ├── hyperparameters.yaml           # Hyperparameters for the model
│
├── src/
│   ├── training/
│   │   ├── local_training.py          # Local training script
│   │   ├── distributed_training.py    # Distributed training script (Ray.io-based)
│   │   ├── utils.py                   # Utility functions for training
│   │   ├── model.py                   # Model architecture (RNN, CNN, etc.)
│   │   ├── trainer.py                 # Training loop logic for Ray and PyTorch
│
├── deployment/
│   ├── ray_cluster_setup.sh           # Bash script for setting up the Ray cluster
│   ├── Dockerfile                     # Docker configuration for creating containers
│
├── scripts/
│   ├── run_distributed_training.sh    # Bash script to launch distributed training
│
└── requirements.txt                   # List of Python dependencies
```

---

## **3. Distributed Training Workflow**

### **3.1 Cluster Setup**
The Ray cluster must be configured before launching any distributed training. Use the provided `ray_cluster_setup.sh` script, which automates the process of starting Ray on multiple machines (head node and workers).

### **3.2 Ray Cluster Configuration**
Configure your Ray cluster in a `cluster.yaml` file. This file includes specifications for both head and worker nodes, such as resource allocation (CPUs/GPUs, memory) and environment setup.

**Sample `cluster.yaml` Configuration:**
```yaml
cluster_name: ray_cluster
min_workers: 2
max_workers: 5

head_node:
  InstanceType: m5.xlarge
  ImageId: ami-0123456789abcdefg
  KeyName: your-keypair

worker_nodes:
  InstanceType: g4dn.xlarge
  ImageId: ami-0123456789abcdefg
  KeyName: your-keypair

setup_commands:
  - pip install -r /path/to/requirements.txt

head_node_options:
  include_dashboard: True
  dashboard_port: 8265

file_mounts:
  /path/to/project_root: /home/ubuntu/project_root

auth:
  ssh_user: ubuntu
```

**Steps:**
1. **Head Node**: Launch the head node of the Ray cluster by running:
   ```bash
   ray up cluster.yaml --head
   ```
   This starts the Ray head node, which manages tasks and communication between worker nodes.

2. **Worker Nodes**: Launch worker nodes to join the cluster:
   ```bash
   ray up cluster.yaml
   ```
   The number of workers can be scaled based on the resource requirements of your distributed training task.

---

## **4. Distributed Training Script (`distributed_training.py`)**

The core of distributed training is implemented in `distributed_training.py`, where **Ray.io** and **PyTorch** are used to parallelize the training process across nodes.

```python
import os
import ray
import torch
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from src.models.model import RNNModel  # Example model
from src.training.trainer import Trainer
from src.utils.data_loader import load_data
from src.utils.checkpoint_utils import save_checkpoint

# Initialize Ray
ray.init(address="auto")

# Load hyperparameters from YAML
import yaml
with open("/path/to/config/hyperparameters.yaml", 'r') as f:
    hyperparams = yaml.safe_load(f)

with open("/path/to/config/environment.yaml", 'r') as f:
    env_config = yaml.safe_load(f)

# Set up data
train_loader, val_loader, test_loader = load_data(
    env_config['paths']['processed_data_dir'],
    hyperparams['training']['batch_size']
)

# Training function for each worker
def train_fn(config):
    model = RNNModel(
        vocab_size=hyperparams['dataset']['vocab_size'],
        embedding_dim=hyperparams['model']['embedding_dim'],
        hidden_dim=hyperparams['model']['hidden_dim'],
        output_dim=4,  # Assuming 4 classes
        n_layers=hyperparams['model']['n_layers'],
        bidirectional=hyperparams['model']['bidirectional'],
        dropout=hyperparams['model']['dropout']
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer initialization
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders={"train": train_loader, "val": val_loader, "test": test_loader},
        save_dir=env_config["paths"]["model_save_dir"]
    )
    
    # Start training
    trainer.train(num_epochs=config["epochs"])

# ScalingConfig for Ray
scaling_config = ScalingConfig(
    num_workers=4,  # Number of parallel workers
    use_gpu=torch.cuda.is_available(),  # Use GPU if available
    resources_per_worker={"GPU": 1 if torch.cuda.is_available() else 0}  # Allocate GPUs
)

# Set up the trainer in Ray
trainer = TorchTrainer(
    train_loop_per_worker=train_fn,
    scaling_config=scaling_config,
    datasets={"train": train_loader},
    config={"lr": tune.uniform(1e-4, 1e-2), "epochs": tune.choice([5, 10, 20])}
)

# Launch hyperparameter search
tuner = tune.Tuner(
    trainer,
    param_space={"lr": tune.uniform(1e-4, 1e-2), "epochs": tune.choice([5, 10, 20])},
    tune_config=tune.TuneConfig(num_samples=10, metric="accuracy", mode="max")
)

# Execute the training
results = tuner.fit()

# Get best trial
best_result = results.get_best_result(metric="accuracy", mode="max")
print("Best result config: ", best_result.config)
```

---

## **5. Running Distributed Training**

To run distributed training on the cluster, follow these steps:

### **5.1 Set Up Ray Cluster**
Use the `ray_cluster_setup.sh` script located in the `/deployment` directory to start the cluster:
```bash
bash deployment/ray_cluster_setup.sh
```

### **5.2 Launch Distributed Training**
Once the cluster is running, use the `run_distributed_training.sh` script to start the distributed training job across the cluster:
```bash
bash scripts/run_distributed_training.sh
```

This script internally runs the `distributed_training.py` script using Ray to distribute the workload across nodes.

---

## **6. Monitoring and Debugging**

### **6.1 Ray Dashboard**
Ray provides a built-in dashboard that allows you to monitor tasks, resource usage (CPU/GPU), and worker status in real-time.

To access the dashboard, navigate to the Ray head node IP and port `8265`:
```
http://<head_node_ip>:8265
```

### **6.2 Logs**
Ensure logging is configured using **structured logs** (e.g., `logging_utils.py`). Logs are stored locally on each node and provide insights into each worker’s progress.

---

## **7. Best Practices for Distributed Training**

1. **Optimizing Data Loading**: Ensure the dataset is pre-sharded and each worker processes a different shard of the data. This can significantly speed up training when working with large datasets.
   
2. **Load Balancing**: Properly configure the number of workers to match the GPU/CPU resources available across your cluster.

3. **Checkpointing**: Regularly checkpoint models during training to handle potential node failures and ensure that the training can resume from the last checkpoint.

4. **Mixed Precision Training**: Use mixed precision training (FP16) to accelerate training and reduce memory usage, especially on GPUs.

5. **Auto-scaling**: Configure Ray to automatically scale the cluster up or down based on resource usage to optimize costs on cloud platforms.