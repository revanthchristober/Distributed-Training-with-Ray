import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report

# For advanced plots, using seaborn and matplotlib
sns.set(style="whitegrid")


def visualize_model_architecture(model, input_size, device):
    """
    Visualize and print the model's architecture.

    Parameters:
    - model (torch.nn.Module): The PyTorch model instance.
    - input_size (tuple): The input dimensions of the model, e.g., (1, 28, 28) for image data.
    - device (torch.device): The device to use for model summary visualization.
    
    Returns:
    - None
    """
    model.to(device)
    print("Model Architecture:")
    print(summary(model, input_size=input_size))


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix using seaborn for better visualization.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - class_names (list): List of class names for labeling the axes.

    Returns:
    - None
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_classification_report(y_true, y_pred, class_names):
    """
    Display classification report metrics (precision, recall, F1-score) as a heatmap.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - class_names (list): List of class names.

    Returns:
    - None
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title('Classification Report')
    plt.show()


def plot_training_history(training_history):
    """
    Plot the training and validation accuracy/loss curves.

    Parameters:
    - training_history (dict): A dictionary with keys "train_loss", "val_loss", "train_acc", "val_acc".

    Returns:
    - None
    """
    epochs = range(1, len(training_history['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_history['train_loss'], label='Training Loss')
    plt.plot(epochs, training_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, training_history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_precision_recall_f1_per_class(y_true, y_pred, class_names):
    """
    Plot bar charts for precision, recall, and F1-score for each class.

    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - class_names (list): List of class names for labeling the x-axis.

    Returns:
    - None
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extracting the precision, recall, f1 for each class
    precision = [report[cls]['precision'] for cls in class_names]
    recall = [report[cls]['recall'] for cls in class_names]
    f1 = [report[cls]['f1-score'] for cls in class_names]

    x = np.arange(len(class_names))

    # Bar chart for precision, recall, F1-score
    plt.figure(figsize=(12, 7))
    width = 0.2
    plt.bar(x - width, precision, width=width, label='Precision', color='b')
    plt.bar(x, recall, width=width, label='Recall', color='g')
    plt.bar(x + width, f1, width=width, label='F1-score', color='r')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, F1-Score per Class')
    plt.xticks(x, class_names)
    plt.legend()
    plt.show()


def save_plots_to_file(plot_func, file_path, *args, **kwargs):
    """
    Save a plot to a file by wrapping the plotting function.

    Parameters:
    - plot_func (function): Plotting function to wrap.
    - file_path (str): Path where the plot will be saved.
    - args, kwargs: Additional arguments for the plotting function.

    Returns:
    - None
    """
    plot_func(*args, **kwargs)
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")


def visualize_model_gradients(model, input_tensor):
    """
    Visualize model gradients using a heatmap (for convolutional layers).

    Parameters:
    - model (torch.nn.Module): PyTorch model.
    - input_tensor (torch.Tensor): Input tensor for which gradients are computed.

    Returns:
    - None
    """
    # Enable gradients for the input
    input_tensor.requires_grad_()
    
    # Forward pass
    model.eval()
    output = model(input_tensor)
    output_idx = output.argmax()
    
    # Backward pass
    model.zero_grad()
    output[0, output_idx].backward()

    # Get the gradients for the input tensor
    gradients = input_tensor.grad.data.cpu().numpy().squeeze()

    plt.figure(figsize=(8, 6))
    sns.heatmap(gradients, cmap='coolwarm', center=0)
    plt.title("Gradient Heatmap")
    plt.show()


# Example usage:
if __name__ == "__main__":
    import torch
    from src.models.rnn_model import RNNModel  # Assuming you have an RNN model defined in your project

    # Mock data and model for the example
    mock_model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=256, output_dim=4, n_layers=2, bidirectional=True, dropout=0.5, rnn_type='LSTM')
    mock_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock data
    y_true = np.array([0, 1, 2, 1, 0, 2, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 2])
    class_names = ['Class 0', 'Class 1', 'Class 2']

    # Visualize the architecture
    visualize_model_architecture(mock_model, (1, 128), mock_device)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

    # Plot classification report
    plot_classification_report(y_true, y_pred, class_names)

    # Plot precision, recall, F1-score per class
    plot_precision_recall_f1_per_class(y_true, y_pred, class_names)

    # Mock training history data
    training_history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3],
        "val_loss": [0.9, 0.7, 0.5, 0.4],
        "train_acc": [60, 70, 80, 85],
        "val_acc": [55, 65, 75, 80]
    }

    # Plot training history
    plot_training_history(training_history)
