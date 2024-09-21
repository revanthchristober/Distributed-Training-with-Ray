import ray
import pytest
import torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from src.models.rnn_model import RNNModel
from src.training.trainer import Trainer
from src.utils.data_loader import load_data
import yaml

@pytest.fixture(scope="module")
def ray_cluster():
    """
    Fixture to set up a Ray cluster for distributed training tests.
    """
    ray.init(num_cpus=8)  # Adjust the number of CPUs based on your environment
    yield
    ray.shutdown()


def test_distributed_training(ray_cluster):
    """
    Test to ensure distributed training works as expected across multiple nodes.
    """
    # Load configuration
    with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/hyperparameteres.yaml", 'r') as f:
        hyperparams = yaml.safe_load(f)

    with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/environment.yaml", 'r') as f:
        env_config = yaml.safe_load(f)

    # Set device (CPU/GPU)
    device = torch.device(f"cuda:{env_config['device']['gpu_id']}" if env_config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, val_loader, _ = load_data(env_config['paths']['processed_data_dir'], hyperparams['training']['batch_size'])

    # Model Initialization
    model = RNNModel(
        vocab_size=hyperparams['dataset']['vocab_size'],
        embedding_dim=hyperparams['model']['embedding_dim'],
        hidden_dim=hyperparams['model']['hidden_dim'],
        output_dim=4,  # Assuming 4 classes for AG News
        n_layers=hyperparams['model']['n_layers'],
        bidirectional=hyperparams['model']['bidirectional'],
        dropout=hyperparams['model']['dropout'],
        rnn_type=hyperparams['model']['rnn_type']
    ).to(device)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=hyperparams['training']['learning_rate']),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        dataloaders={"train": train_loader, "val": val_loader},
        save_dir=env_config["paths"]["model_save_dir"],
        model_name="rnn_model_test"
    )

    # Define the distributed scaling config for Ray
    scaling_config = ScalingConfig(num_workers=4)  # Adjust number of workers based on environment

    # Initialize Ray Trainer
    ray_trainer = TorchTrainer(
        train_loop_per_worker=trainer.train_epoch,
        scaling_config=scaling_config
    )

    # Execute Training
    result = ray_trainer.fit()

    # Assertions for Distributed Training
    assert result is not None, "Distributed training failed, result is None."
    assert isinstance(result.metrics, dict), "Training result metrics should be a dictionary."
    assert "train_loss" in result.metrics, "train_loss should be in training metrics."
    assert result.metrics["train_loss"] < 1.0, "Training loss should be below 1.0 for successful training."

