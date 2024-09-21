import pytest
import torch
from src.models.rnn_model import RNNModel

@pytest.fixture
def model():
    """
    Fixture to initialize the model with predefined hyperparameters.
    """
    return RNNModel(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=4,  # Assuming 4 classes
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        rnn_type="LSTM"
    ).to(torch.device('cpu'))

def test_model_forward_pass(model):
    """
    Test the forward pass of the model.
    """
    sample_input = torch.randint(0, 10000, (32, 50))  # Batch of 32, Sequence length of 50
    output = model(sample_input)
    
    assert output.shape == (32, 4), "Model output should match the (batch_size, output_dim) shape."
    assert torch.is_tensor(output), "Model output should be a torch tensor."
    assert output.requires_grad, "Model output should require gradients."

def test_embedding_layer(model):
    """
    Test the embedding layer of the model.
    """
    sample_input = torch.randint(0, 10000, (10, 20))  # Batch of 10, Sequence length of 20
    embedding_output = model.embedding(sample_input)
    
    assert embedding_output.shape == (10, 20, 128), "Embedding output should match the (batch_size, seq_length, embedding_dim) shape."

def test_bidirectional_rnn(model):
    """
    Test if the RNN is bidirectional.
    """
    assert model.rnn.bidirectional, "RNN should be bidirectional."
