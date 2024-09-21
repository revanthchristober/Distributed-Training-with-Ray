import pytest
import torch
import os
import yaml
from src.models.rnn_model import RNNModel
from src.training.trainer import Trainer
from src.utils.data_loader import load_data
from src.evaluation.evaluate_model import evaluate
from src.utils.checkpoint_utils import load_checkpoint

@pytest.fixture(scope="module")
def config():
    """
    Fixture to load the configuration files before running tests.
    """
    with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/hyperparameteres.yaml", 'r') as f:
        hyperparams = yaml.safe_load(f)

    with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/environment.yaml", 'r') as f:
        env_config = yaml.safe_load(f)

    return hyperparams, env_config


@pytest.fixture(scope="module")
def data_loaders(config):
    """
    Fixture to set up and return data loaders for the pipeline.
    """
    hyperparams, env_config = config
    train_loader, val_loader, test_loader = load_data(env_config['paths']['processed_data_dir'], hyperparams['training']['batch_size'])
    return train_loader, val_loader, test_loader


@pytest.fixture(scope="module")
def model_and_trainer(config, data_loaders):
    """
    Fixture to set up the model and trainer for the pipeline.
    """
    train_loader, val_loader, _ = data_loaders
    hyperparams, env_config = config

    # Set device (CPU/GPU)
    device = torch.device(f"cuda:{env_config['device']['gpu_id']}" if env_config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

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
        model_name="rnn_model_pipeline_test"
    )

    return model, trainer


def test_data_loading(data_loaders):
    """
    Test to ensure data is loaded correctly and in the expected format.
    """
    train_loader, val_loader, test_loader = data_loaders

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Check if the data is in the expected format (e.g., tensors)
    sample_data, sample_label = next(iter(train_loader))
    assert isinstance(sample_data, torch.Tensor), "Data should be a torch Tensor."
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch Tensor."


def test_training(model_and_trainer, config):
    """
    Test to ensure the model trains correctly and checkpoints are saved.
    """
    _, trainer = model_and_trainer
    hyperparams, env_config = config

    # Train the model (run only 2 epochs for testing purposes)
    trainer.train(num_epochs=2)

    # Check if model checkpoint is saved
    model_file_path = os.path.join(env_config["paths"]["model_save_dir"], "rnn_model_pipeline_test.pt")
    assert os.path.exists(model_file_path), "Model checkpoint should be saved after training."

    # Ensure model checkpoint can be loaded successfully
    loaded_model = load_checkpoint(model_file_path)
    assert loaded_model is not None, "Model checkpoint should load correctly."

    # Clean up: Remove the test checkpoint after testing
    if os.path.exists(model_file_path):
        os.remove(model_file_path)


def test_evaluation(model_and_trainer, data_loaders):
    """
    Test to ensure the evaluation on test data works as expected.
    """
    model, _ = model_and_trainer
    _, _, test_loader = data_loaders

    # Set device (CPU/GPU)
    device = next(model.parameters()).device

    # Evaluate the model on test data
    test_loss, test_accuracy = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)

    # Assertions for evaluation
    assert test_loss < 1.5, f"Test loss is too high: {test_loss}"
    assert test_accuracy > 70, f"Test accuracy is too low: {test_accuracy}%"


def test_pipeline(model_and_trainer, data_loaders, config):
    """
    Test the entire pipeline (data loading, training, evaluation).
    """
    model, trainer = model_and_trainer
    train_loader, val_loader, test_loader = data_loaders
    hyperparams, env_config = config

    # Ensure data is loaded properly
    test_data_loading(data_loaders)

    # Train the model (for 2 epochs)
    test_training(model_and_trainer, config)

    # Evaluate the model on test data
    test_evaluation(model_and_trainer, data_loaders)
