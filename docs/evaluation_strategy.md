# **Evaluation Strategy for Model Performance**

## **1. Overview**

Evaluating the performance of machine learning models is crucial for understanding how well they generalize to unseen data. In this guide, we explain the evaluation strategy used in our project, focusing on both classification and regression tasks. We will cover common metrics, advanced techniques, and how these metrics are computed and tracked during and after training.

This document will cover:

1. Evaluation during training (validation metrics)
2. Post-training evaluation on the test dataset
3. Custom performance metrics
4. Model comparison strategies
5. Visualization techniques for interpreting results

---

## **2. Evaluation Metrics**

### **2.1 Classification Metrics**
For classification tasks, a variety of metrics are used to evaluate model performance. Depending on the dataset and the problem type, one or more of these metrics are computed:

#### **Accuracy**
**Formula**:  
\[
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
\]
Accuracy measures the proportion of correctly classified instances out of all predictions made.

#### **Precision, Recall, and F1-Score**
These metrics are essential when the dataset is imbalanced.

- **Precision**: Measures the proportion of true positive results out of all predicted positives.
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
\]

- **Recall (Sensitivity)**: Measures the proportion of true positive results out of all actual positives.
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]

- **F1-Score**: Harmonic mean of Precision and Recall, balancing both metrics.
\[
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

#### **Confusion Matrix**
The confusion matrix shows how many instances were correctly and incorrectly predicted for each class. For multi-class classification problems, the confusion matrix is a powerful visualization tool.

**Confusion Matrix Example for 3 classes**:
| Actual/Predicted | Class 1 | Class 2 | Class 3 |
|------------------|---------|---------|---------|
| **Class 1**       | 50      | 10      | 0       |
| **Class 2**       | 5       | 40      | 10      |
| **Class 3**       | 0       | 5       | 60      |

#### **ROC Curve and AUC (Area Under the Curve)**
- The **ROC (Receiver Operating Characteristic) curve** plots the true positive rate (TPR) against the false positive rate (FPR).
- **AUC (Area Under Curve)** measures the entire two-dimensional area under the ROC curve. A higher AUC indicates better model performance.

#### **Log Loss (Cross-Entropy Loss)**
Used for evaluating probabilistic classification models.
\[
\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
\]

### **2.2 Regression Metrics**
For regression tasks, common evaluation metrics include:

#### **Mean Squared Error (MSE)**
**Formula**:  
\[
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]
MSE measures the average squared difference between the actual and predicted values.

#### **Root Mean Squared Error (RMSE)**
**Formula**:  
\[
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
\]
RMSE is the square root of MSE, which makes it interpretable in the same units as the output variable.

#### **Mean Absolute Error (MAE)**
**Formula**:  
\[
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|
\]
MAE measures the average of the absolute differences between the actual and predicted values.

#### **R-Squared (R²)**
R-squared represents the proportion of variance in the dependent variable that is predictable from the independent variables.
\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]
Where \( \bar{y} \) is the mean of the actual values.

---

## **3. Evaluation Strategy During Training**

During training, models are evaluated at the end of each epoch using a validation set to monitor overfitting and to tune hyperparameters. Validation metrics include accuracy, precision, recall, and loss.

### **3.1 Validation Frequency**
- **Validation at every epoch**: After each training epoch, the model is evaluated on the validation set, and performance metrics are recorded.
- **Early stopping**: If the validation metrics stop improving for a defined number of epochs (patience), training is terminated early to prevent overfitting.

### **3.2 Hyperparameter Tuning**
Metrics like validation loss, accuracy, and F1-score are used to optimize hyperparameters during tuning. **Ray Tune** is used for hyperparameter optimization, leveraging metrics to find the best performing configurations.

---

## **4. Post-Training Evaluation**

Once the model has been trained, the final evaluation is performed on the test dataset, which has been kept unseen during training.

### **4.1 Test Set Evaluation**
After training, the model is evaluated on the test set using the following steps:
- **Predictions** are made for the test set.
- Metrics such as accuracy, precision, recall, and F1-score are computed.
- **Confusion matrix** is generated to visualize classification performance.

### **4.2 Model Checkpoint Evaluation**
Evaluate different model checkpoints to compare model performance across different epochs. The best model is selected based on the highest performance metrics.

---

## **5. Custom Performance Metrics**

Custom metrics can be implemented as per the requirements of the specific use case. Examples include:
- **Balanced Accuracy** for imbalanced datasets:
\[
\text{Balanced Accuracy} = \frac{1}{2}\left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)
\]

- **Jaccard Index** for multi-class classification:
\[
\text{Jaccard Index} = \frac{|A \cap B|}{|A \cup B|}
\]
Where \( A \) and \( B \) are the predicted and true sets, respectively.

- **Custom Loss Functions**: Implement custom loss functions to fit specific business goals, such as profit-weighted loss in financial models.

---

## **6. Model Comparison**

After training multiple models (e.g., during hyperparameter tuning), performance across models is compared using key metrics like accuracy, F1-score, and AUC. The comparison is made based on:
- **Test set performance**: Use test set metrics to compare models.
- **Training time**: Balance between model performance and training efficiency.
- **Model complexity**: Choose models with lower complexity but high performance to avoid overfitting.

### **6.1 Cross-Validation**
In addition to test set evaluation, cross-validation can be used to assess the model's ability to generalize to different data subsets.

---

## **7. Visualization of Model Performance**

Visualizing model performance helps in interpreting how well the model works and where it struggles. Some of the key visualizations include:

### **7.1 Loss Curves**
Plot training and validation loss curves to track model convergence and detect overfitting.

### **7.2 ROC Curves**
Plot the ROC curve to visualize the trade-off between TPR and FPR. The area under the curve (AUC) provides a single-value summary of model performance.

### **7.3 Precision-Recall Curves**
This is particularly useful when dealing with imbalanced datasets. It helps visualize how precision and recall change with different threshold values.

### **7.4 Confusion Matrix**
Generate a heatmap of the confusion matrix to visualize the model's performance for each class, especially highlighting where it confuses between classes.

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

### **7.5 Calibration Plots**
For probabilistic models, calibration curves can show how well the predicted probabilities match actual outcomes.

---

## **8. Tracking Model Performance**

### **8.1 Logging and Checkpointing**
Structured logging of metrics at each epoch is essential for tracking progress and debugging. Use custom logging utilities to record metrics such

 as loss, accuracy, precision, and recall after each epoch.

### **8.2 Model Checkpointing**
Save the model checkpoint with the best validation metrics (e.g., lowest validation loss or highest validation accuracy). This allows the model to be restored and used for future inference or fine-tuning.

```python
# Example of saving a checkpoint after training
torch.save(model.state_dict(), "best_model.pth")
```

---

## **9. Best Practices**

1. **Metric Selection**: Choose evaluation metrics that align with the business goals. For example, use F1-score for imbalanced datasets and accuracy for balanced datasets.
2. **Data Splitting**: Always evaluate on unseen test data to get an unbiased estimate of model performance.
3. **Hyperparameter Optimization**: Use cross-validation and validation metrics during hyperparameter tuning to ensure the model generalizes well.
4. **Balanced Metrics**: Use precision, recall, and F1-score in combination to ensure that the model performs well in all aspects.