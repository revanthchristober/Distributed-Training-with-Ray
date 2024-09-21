# checkpoint_utils.py

import torch
import os
import logging

logger = logging.getLogger('distributed_training.utils.checkpoint_utils')

def save_checkpoint(model, optimizer, file_path, epoch=None, best_accuracy=None):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        file_path (str): The file path where the checkpoint will be saved.
        epoch (int, optional): The current epoch (useful for resuming training).
        best_accuracy (float, optional): The best validation accuracy so far.
    """
    try:
        logger.info(f"Saving checkpoint to {file_path}...")
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
            logger.debug(f"Epoch saved in checkpoint: {epoch}")
        if best_accuracy is not None:
            checkpoint['best_accuracy'] = best_accuracy
            logger.debug(f"Best validation accuracy saved in checkpoint: {best_accuracy:.4f}")
        
        # Save checkpoint to the file
        torch.save(checkpoint, file_path)
        logger.info(f"Checkpoint saved successfully at {file_path}.")
    except Exception as e:
        logger.error(f"Error while saving checkpoint: {e}")
        raise

def load_checkpoint(model, optimizer, file_path, device):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        file_path (str): The file path to load the checkpoint from.
        device (torch.device): The device to load the model onto (CPU/GPU).

    Returns:
        tuple: The loaded epoch (if available) and best accuracy (if available).
    """
    if not os.path.isfile(file_path):
        logger.error(f"No checkpoint file found at {file_path}.")
        raise FileNotFoundError(f"No checkpoint file found at {file_path}.")
    
    logger.info(f"Loading checkpoint from {file_path}...")
    try:
        # Load the checkpoint
        checkpoint = torch.load(file_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded from checkpoint.")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded from checkpoint.")

        # Load epoch and best accuracy (if available)
        epoch = checkpoint.get('epoch', None)
        best_accuracy = checkpoint.get('best_accuracy', None)

        if epoch is not None:
            logger.info(f"Resuming from epoch: {epoch}")
        if best_accuracy is not None:
            logger.info(f"Best validation accuracy: {best_accuracy:.4f}")

        return epoch, best_accuracy
    except Exception as e:
        logger.error(f"Error while loading checkpoint: {e}")
        raise

def save_best_model(model, file_path):
    """
    Save only the model's state dictionary (used for saving the best model after training).

    Args:
        model (torch.nn.Module): The model to save.
        file_path (str): The file path to save the best model.
    """
    try:
        logger.info(f"Saving best model to {file_path}...")
        torch.save(model.state_dict(), file_path)
        logger.info(f"Best model saved successfully at {file_path}.")
    except Exception as e:
        logger.error(f"Error while saving best model: {e}")
        raise

def load_model(model, file_path, device):
    """
    Load the model's state dictionary from a saved file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        file_path (str): The file path to load the model state from.
        device (torch.device): The device to load the model onto (CPU/GPU).

    Returns:
        torch.nn.Module: The model with the loaded state.
    """
    if not os.path.isfile(file_path):
        logger.error(f"No model file found at {file_path}.")
        raise FileNotFoundError(f"No model file found at {file_path}.")
    
    logger.info(f"Loading model from {file_path}...")
    try:
        # Load model state dictionary
        model.load_state_dict(torch.load(file_path, map_location=device))
        model.to(device)
        logger.info(f"Model loaded successfully from {file_path} and moved to {device}.")
        return model
    except Exception as e:
        logger.error(f"Error while loading model: {e}")
        raise