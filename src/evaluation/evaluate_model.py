import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.models.rnn_model import RNNModel  # Import your model
from src.utils.data_loader import load_data
from src.utils.checkpoint_utils import load_checkpoint
import yaml
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger('evaluation.evaluate_model')

# Load configurations
with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/hyperparameteres.yaml", 'r') as f:
    hyperparams = yaml.safe_load(f)

with open("/workspaces/codespaces-jupyter/Distributed-Model-Training/config/environment.yaml", 'r') as f:
    env_config = yaml.safe_load(f)

# Device configuration (CPU/GPU)
device = torch.device(f"cuda:{env_config['device']['gpu_id']}" if env_config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
logger.info(f"Evaluating model on device: {device}")

# Function to evaluate model performance on test dataset
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss
            total_loss += loss.item()

            # Get predictions and accumulate them
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    # Detailed classification report (per-class metrics)
    class_report = classification_report(all_labels, all_preds, target_names=hyperparams['dataset']['class_names'])

    return avg_loss, accuracy, precision, recall, f1, class_report

# Main function to load model, data, and run evaluation
def run_evaluation():
    # Load the test data
    _, _, test_loader = load_data(
        data_dir=env_config['paths']['processed_data_dir'],
        batch_size=hyperparams['evaluation']['batch_size']
    )

    # Initialize the model (same as in training)
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

    # Load model checkpoint
    checkpoint_path = os.path.join(env_config['paths']['checkpoint_dir'], env_config['evaluation']['checkpoint_file'])
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    model_state, _ = load_checkpoint(checkpoint_path)
    model.load_state_dict(model_state)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    logger.info("Evaluating model performance on test data...")
    avg_loss, accuracy, precision, recall, f1, class_report = evaluate_model(model, test_loader, criterion)

    # Log and display results
    logger.info(f"Average Loss: {avg_loss}")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")
    logger.info("Classification Report:")
    logger.info(f"\n{class_report}")

    print(f"Average Loss: {avg_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    run_evaluation()
