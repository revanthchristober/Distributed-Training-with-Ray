import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import sys
import os
import pickle
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from src.models.rnn_model import RNNModel
from src.training.trainer import Trainer
from src.utils.data_loader import load_data
from src.utils.checkpoint_utils import save_checkpoint
from torch.optim import Adam
import yaml
import logging
import spacy  # Assuming SpaCy is used for tokenization

# Initialize logger
logger = logging.getLogger('distributed_training.training.local_training')

# Vocabulary helper functions
def save_vocab(vocab, vocab_path):
    """Save vocabulary to a file."""
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    logger.info(f"Vocabulary saved at {vocab_path}")

def load_vocab(vocab_path):
    """Load vocabulary from a file."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    logger.info(f"Vocabulary loaded from {vocab_path}")
    return vocab

def build_vocabulary(data, tokenizer, vocab_size, vocab_path):
    """Build or load vocabulary."""
    if os.path.exists(vocab_path):
        logger.info("Vocabulary file found. Loading vocabulary...")
        vocab = load_vocab(vocab_path)
    else:
        logger.info("Vocabulary file not found. Building vocabulary...")
        vocab = {}
        for idx, row in data.iterrows():
            tokens = tokenizer(str(row['Title']) + " " + str(row['Description']))
            for token in tokens:
                vocab[token.text] = vocab.get(token.text, 0) + 1

        # Sort and limit the vocab size
        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:vocab_size]}
        
        # Add special tokens
        vocab["<UNK>"] = len(vocab)
        vocab["<PAD>"] = len(vocab) + 1

        save_vocab(vocab, vocab_path)

    return vocab

# Load hyperparameters from YAML files
with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/hyperparameteres.yaml", 'r') as f:
    hyperparams = yaml.safe_load(f)

with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/environment.yaml", 'r') as f:
    env_config = yaml.safe_load(f)

# Set device (CPU/GPU)
device = torch.device(f"cuda:{env_config['device']['gpu_id']}" if env_config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize SpaCy tokenizer
nlp = spacy.load("en_core_web_sm")

# Vocabulary setup
vocab_path = os.path.join(env_config['paths']['vocab_dir'], 'vocab.pkl')
vocab_size = hyperparams['dataset']['vocab_size']

# Assuming the data is already loaded as pandas DataFrame
# train_data, val_data, test_data = load_data(env_config['paths']['processed_data_dir'])
train_loader, val_loader, test_loader = load_data(
    env_config['paths']['processed_data_dir'], 
    hyperparams['training']['batch_size']  # Pass the batch size here
)

# Build or load vocabulary
vocab = build_vocabulary(
    data=train_loader,  # Use the training data to build vocabulary
    tokenizer=nlp,    # Use SpaCy tokenizer
    vocab_size=vocab_size,
    vocab_path=vocab_path
)

# Put large objects in Ray's object store
train_loader_id = ray.put(train_loader)
val_loader_id = ray.put(val_loader)
test_loader_id = ray.put(test_loader)

# Define the train function for Ray Tune
def train_loop_per_worker(config):
    # Model initialization
    model = RNNModel(
        vocab_size=len(vocab),
        embedding_dim=hyperparams['model']['embedding_dim'],
        hidden_dim=hyperparams['model']['hidden_dim'],
        output_dim=4,  # Assuming 4 classes for AG News
        n_layers=hyperparams['model']['n_layers'],
        bidirectional=hyperparams['model']['bidirectional'],
        dropout=hyperparams['model']['dropout'],
        rnn_type=hyperparams['model']['rnn_type']
    ).to(device)
    
    # Optimizer and Loss function
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Retrieve DataLoader objects from Ray object store
    train_loader = ray.get(train_loader_id)
    val_loader = ray.get(val_loader_id)
    test_loader = ray.get(test_loader_id)

    # Trainer setup
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        dataloaders={"train": train_loader, "val": val_loader, "test": test_loader},
        save_dir=env_config["paths"]["model_save_dir"],
        model_name="rnn_model_ray"
    )
    
    # Train the model
    trainer.train(num_epochs=config['epochs'])

# Hyperparameter search space
search_space = {
    "learning_rate": tune.uniform(1e-4, 1e-2),
    "epochs": tune.choice([5, 10, 20])
}

# Update the tuning setup to include a metric
tuner = tune.Tuner(
    trainable=train_loop_per_worker,  # Pass the train function as 'trainable'
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=10,  # Number of trials
        metric="accuracy",  # Track accuracy
        mode="max"  # Maximize the accuracy
    ),
)

# Execute the tuning
result_grid = tuner.fit()

# Retrieve the best configuration and result based on accuracy
best_result = result_grid.get_best_result(metric="accuracy", mode="max")
best_config = best_result.config

logger.info(f"Best config: {best_config}")