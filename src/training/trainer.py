# trainer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from tqdm import tqdm
from src.utils.checkpoint_utils import save_checkpoint
from ray import tune

# Initialize logger
logger = logging.getLogger('distributed_training.training.trainer')

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloaders, optimizer, criterion, device, save_dir=None, model_name="model"):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.dataloaders['train'], desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epoch_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
        return epoch_loss, epoch_accuracy

    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['val'], desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epoch_accuracy = 100. * correct / total
        print(f"Validation: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
        return epoch_loss, epoch_accuracy

    def save_model(self):
        """Save the model to the specified directory."""
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            save_path = os.path.join(self.save_dir, f"{self.model_name}.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            print("Save directory not specified. Model not saved.")

    def train(self, num_epochs):
        """Train the model for a given number of epochs."""
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate()
            print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save the model at the end of training
        self.save_model()