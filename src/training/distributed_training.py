import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import ray
from ray import tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from src.models.rnn_model import RNNModel  # Replace with your model
from src.utils.data_loader import load_data, CustomDataset
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from src.training.trainer import Trainer
import yaml

# Initialize logger
logger = logging.getLogger('distributed_training.training.distributed_training')

# Load configurations
with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/hyperparameteres.yaml", 'r') as f:
    hyperparams = yaml.safe_load(f)

with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/environment.yaml", 'r') as f:
    env_config = yaml.safe_load(f)

# Device configuration (CPU/GPU)
device = torch.device(f"cuda:{env_config['device']['gpu_id']}" if env_config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load and process data using Ray Dataset (for distributed training)
def get_datasets():
    train_loader, val_loader, test_loader = load_data(
        data_dir=env_config['paths']['processed_data_dir'],
        batch_size=hyperparams['training']['batch_size'],
    )
    return train_loader, val_loader, test_loader

# Distributed training function
def train_distributed(config, checkpoint_dir=None):
    logger.info("Starting distributed training...")

    # Load datasets
    train_loader, val_loader, test_loader = get_datasets()

    # Initialize model
    model = RNNModel(
        vocab_size=hyperparams['dataset']['vocab_size'],
        embedding_dim=hyperparams['model']['embedding_dim'],
        hidden_dim=hyperparams['model']['hidden_dim'],
        output_dim=4,  # Assuming a 4-class classification problem (e.g., AG News)
        n_layers=hyperparams['model']['n_layers'],
        bidirectional=hyperparams['model']['bidirectional'],
        dropout=hyperparams['model']['dropout'],
        rnn_type=hyperparams['model']['rnn_type']
    ).to(device)

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if available
    if checkpoint_dir:
        model_state, optimizer_state = load_checkpoint(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        dataloaders={"train": train_loader, "val": val_loader, "test": test_loader},
        save_dir=env_config["paths"]["model_save_dir"],
        model_name="distributed_rnn_model"
    )

    # Train the model
    trainer.train(num_epochs=config["epochs"])

    # Save the model checkpoint after training
    save_checkpoint(model, optimizer, env_config["paths"]["checkpoint_dir"], f"model_{tune.get_trial_id()}")

# Ray Tune hyperparameter tuning setup
def distributed_tuning():
    # Search space for hyperparameter tuning
    search_space = {
        "learning_rate": tune.uniform(1e-4, 1e-2),
        "epochs": tune.choice([5, 10, 20])
    }

    # Scheduler for early stopping and resource efficiency
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=20,
        grace_period=1,
        reduction_factor=2
    )

    # Reporter to display progress
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    # Ray Tune configuration for distributed training
    tuner = tune.Tuner(
        trainable=train_distributed,  # Distributed training function
        param_space=search_space,  # Hyperparameter search space
        tune_config=tune.TuneConfig(
            num_samples=10,  # Number of hyperparameter samples
            metric="accuracy",  # Metric to optimize
            mode="max",  # Optimize to maximize the metric
        ),
        run_config=tune.RunConfig(
            local_dir=env_config["paths"]["ray_results_dir"],  # Directory to store results
            checkpoint_at_end=True,  # Save checkpoints at the end of each run
            progress_reporter=reporter,  # Display tuning progress
        ),
    )

    # Execute the tuning
    results = tuner.fit()

    # Get the best trial based on accuracy
    best_trial = results.get_best_trial(metric="accuracy", mode="max")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial accuracy: {best_trial.metric_analysis['accuracy']['max']}")

    # Get the best model checkpoint
    best_checkpoint = best_trial.checkpoint
    logger.info(f"Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint, best_trial.config

# Execute the distributed training with hyperparameter tuning
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": os.getcwd()})

    try:
        logger.info("Starting Ray distributed training...")
        best_checkpoint, best_config = distributed_tuning()

        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Best model saved at: {best_checkpoint}")

    finally:
        ray.shutdown()
