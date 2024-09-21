import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    
    Returns:
    - Accuracy score (float).
    """
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, average='weighted', zero_division=1):
    """
    Calculate precision.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - average (str): Type of averaging to be performed, 'micro', 'macro', 'weighted'.
    - zero_division (int): Value to return when encountering zero division.
    
    Returns:
    - Precision score (float).
    """
    return precision_score(y_true, y_pred, average=average, zero_division=zero_division)

def recall(y_true, y_pred, average='weighted', zero_division=1):
    """
    Calculate recall.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - average (str): Type of averaging to be performed, 'micro', 'macro', 'weighted'.
    - zero_division (int): Value to return when encountering zero division.
    
    Returns:
    - Recall score (float).
    """
    return recall_score(y_true, y_pred, average=average, zero_division=zero_division)

def f1(y_true, y_pred, average='weighted', zero_division=1):
    """
    Calculate F1-score.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - average (str): Type of averaging to be performed, 'micro', 'macro', 'weighted'.
    - zero_division (int): Value to return when encountering zero division.
    
    Returns:
    - F1 score (float).
    """
    return f1_score(y_true, y_pred, average=average, zero_division=zero_division)

def confusion_matrix_metrics(y_true, y_pred, labels=None):
    """
    Generate confusion matrix and extract metrics for each class.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - labels (list): List of class labels (optional).
    
    Returns:
    - Confusion matrix (np.array), per-class precision, recall, F1-score (dict).
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_metrics = {
        "precision": precision(y_true, y_pred, average=None, zero_division=1),
        "recall": recall(y_true, y_pred, average=None, zero_division=1),
        "f1": f1(y_true, y_pred, average=None, zero_division=1)
    }
    
    return conf_matrix, class_metrics

def specificity(y_true, y_pred):
    """
    Calculate specificity (true negative rate).
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    
    Returns:
    - Specificity score (float).
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return specificity

def matthews_corrcoef(y_true, y_pred):
    """
    Calculate Matthews correlation coefficient (MCC) for binary classification.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    
    Returns:
    - MCC score (float).
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0

def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy (average of recall for each class).
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    
    Returns:
    - Balanced accuracy score (float).
    """
    recall_per_class = recall(y_true, y_pred, average=None)
    return np.mean(recall_per_class)

def classwise_metrics(y_true, y_pred, class_names):
    """
    Calculate detailed metrics per class: precision, recall, F1-score.
    
    Parameters:
    - y_true (list or np.array): Ground truth labels.
    - y_pred (list or np.array): Predicted labels.
    - class_names (list): List of class names corresponding to class labels.
    
    Returns:
    - Dictionary containing precision, recall, F1-score for each class.
    """
    precision_scores = precision(y_true, y_pred, average=None)
    recall_scores = recall(y_true, y_pred, average=None)
    f1_scores = f1(y_true, y_pred, average=None)
    
    class_metrics = {}
    for idx, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            "precision": precision_scores[idx],
            "recall": recall_scores[idx],
            "f1_score": f1_scores[idx]
        }
    
    return class_metrics

def print_metrics(metrics_dict):
    """
    Print the evaluation metrics in a structured format.
    
    Parameters:
    - metrics_dict (dict): Dictionary containing metric names and their values.
    
    Returns:
    - None
    """
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

# Example usage (assuming y_true and y_pred are numpy arrays)
if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 1, 0, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 0, 2, 1, 2])
    class_names = ["Class 0", "Class 1", "Class 2"]

    metrics = {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall": recall(y_true, y_pred),
        "F1-Score": f1(y_true, y_pred),
        "Specificity": specificity(y_true, y_pred),
        "Matthews Correlation Coefficient": matthews_corrcoef(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy(y_true, y_pred)
    }
    
    classwise = classwise_metrics(y_true, y_pred, class_names)
    print("Overall Metrics:")
    print_metrics(metrics)

    print("\nClass-wise Metrics:")
    for class_name, class_metric in classwise.items():
        print(f"{class_name}: Precision={class_metric['precision']:.4f}, Recall={class_metric['recall']:.4f}, F1-Score={class_metric['f1_score']:.4f}")
