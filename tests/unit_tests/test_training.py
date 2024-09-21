import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.training.trainer import Trainer

@pytest.fixture
def mock_dataloaders():
    """
    Fixture to create mock data loaders for training and validation.
    """
    train_data = [(torch.randn(32, 50), torch.randint(0, 4, (32,)))]  # Batch of 32, seq_len of 50
    val_data = [(torch.randn(32, 50), torch.randint(0, 4, (32,)))]
    return {"train": iter(train_data), "val": iter(val_data)}

@pytest.fixture
def mock_model():
    """
    Fixture to create a mock model.
    """
    model = MagicMock()
    model.return_value = torch.randn(32, 4)  # Mock output shape
    return model

@pytest.fixture
def trainer(mock_model, mock_dataloaders):
    """
    Fixture to create a Trainer instance with mocked components.
    """
    return Trainer(
        model=mock_model,
        optimizer=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        dataloaders=mock_dataloaders,
        save_dir="/tmp",
        model_name="test_model"
    )

def test_training_step(trainer, mock_model):
    """
    Test a single training step.
    """
    mock_model.return_value = torch.randn(32, 4)  # Mock forward pass output
    train_loss, train_acc = trainer.train_epoch(0)
    
    assert isinstance(train_loss, float), "Loss should be a float."
    assert isinstance(train_acc, float), "Accuracy should be a float."
    assert train_loss >= 0, "Loss should be non-negative."
    assert 0 <= train_acc <= 100, "Accuracy should be between 0 and 100."

def test_validation_step(trainer, mock_model):
    """
    Test a single validation step.
    """
    mock_model.return_value = torch.randn(32, 4)  # Mock forward pass output
    val_loss, val_acc = trainer.evaluate_epoch("val")
    
    assert isinstance(val_loss, float), "Validation loss should be a float."
    assert isinstance(val_acc, float), "Validation accuracy should be a float."
    assert val_loss >= 0, "Validation loss should be non-negative."
    assert 0 <= val_acc <= 100, "Validation accuracy should be between 0 and 100."

def test_optimizer_step(trainer):
    """
    Test if the optimizer performs a step.
    """
    trainer.optimizer.step = MagicMock()
    trainer.optimizer.step.assert_not_called()  # Check it's not called before
    trainer.train_epoch(0)
    trainer.optimizer.step.assert_called(), "Optimizer should perform a step during training."
