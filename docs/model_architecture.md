
# **Model Architecture Documentation**

## **Overview**
This document outlines the various deep learning model architectures employed in this project for Natural Language Processing (NLP) tasks, specifically focusing on **Recurrent Neural Networks (RNN)** and **Convolutional Neural Networks (CNN)**. These architectures are used for text classification tasks such as classifying news articles from the **AG News dataset**.

Each architecture is designed to handle different aspects of sequence data, enabling the models to capture temporal dependencies (RNN) and local feature patterns (CNN). The architectures are tuned and optimized for large-scale distributed training using **Ray.io**.

---

## **1. Recurrent Neural Networks (RNN)**

### **1.1 Overview**
Recurrent Neural Networks (RNNs) are a class of neural networks that excel at learning from sequential data. In NLP tasks, RNNs are particularly useful for processing text sequences, as they maintain information across time steps, making them ideal for tasks like text classification, translation, and sequence prediction.

The RNN architecture in this project is designed with flexibility in mind, allowing the choice of different recurrent cells (e.g., **Vanilla RNN**, **LSTM**, **GRU**) depending on the task at hand.

### **1.2 Model Architecture**
```plaintext
+------------------------------------------------+
|                Embedding Layer                 |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|              Recurrent Layers (RNN/LSTM/GRU)   |
|  (Multiple stacked layers with dropout)        |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|                Fully Connected Layer           |
|     (For classification, with softmax output)  |
+------------------------------------------------+
```

- **Embedding Layer**: Converts words into dense vectors of a fixed size (`embedding_dim`). These embeddings capture semantic relationships between words.
- **Recurrent Layer**: The core of the model. Depending on the configuration, this layer can use a Vanilla RNN, LSTM, or GRU. These layers maintain a hidden state that gets updated at each time step, allowing the model to capture long-term dependencies.
  - **LSTM (Long Short-Term Memory)**: Handles long-term dependencies and avoids the vanishing gradient problem by using memory cells and gates (input, forget, and output gates).
  - **GRU (Gated Recurrent Unit)**: A more computationally efficient alternative to LSTM, with fewer gates but similar performance in many tasks.
- **Fully Connected Layer**: Maps the final hidden state to the output classes (e.g., AG News has 4 classes).
- **Dropout**: Prevents overfitting by randomly dropping units from the network during training.

### **1.3 Key Hyperparameters**
- **vocab_size**: Size of the vocabulary used in the embedding layer.
- **embedding_dim**: Dimensionality of the word embeddings (e.g., 100, 300).
- **hidden_dim**: Number of units in each recurrent layer.
- **n_layers**: Number of recurrent layers to stack (e.g., 2–3).
- **dropout**: Dropout rate for regularization (e.g., 0.3).
- **bidirectional**: Whether to use a bidirectional RNN (processes data both forwards and backwards).

### **1.4 Rationale**
RNNs are chosen for their ability to handle sequential data, making them ideal for NLP tasks. The use of LSTM or GRU mitigates the vanishing gradient problem, making them more effective for longer sequences. Bidirectional RNNs enhance performance by considering both past and future context in the sequence.

---

## **2. Convolutional Neural Networks (CNN)**

### **2.1 Overview**
Convolutional Neural Networks (CNNs) are typically used for computer vision tasks, but they have been adapted for text classification. In this project, CNNs are used to capture local patterns in text, such as n-grams, through convolutional filters. CNNs are efficient, highly parallelizable, and have been shown to perform well in text classification tasks.

### **2.2 Model Architecture**
```plaintext
+------------------------------------------------+
|                Embedding Layer                 |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|      1D Convolutional Layer (Multiple Filters)  |
|  (Captures n-grams with varying kernel sizes)   |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|              Max-Pooling Layer                 |
|   (Reduces dimensionality and retains features)|
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|                Fully Connected Layer           |
|     (For classification, with softmax output)  |
+------------------------------------------------+
```

- **Embedding Layer**: Similar to the RNN architecture, this layer converts words into dense vectors.
- **1D Convolutional Layer**: Applies convolutional filters over the text sequence. Each filter is designed to capture specific patterns (n-grams) in the input. Different filter sizes (e.g., 3, 4, 5) are used to capture different lengths of n-grams.
- **Max-Pooling Layer**: Reduces the output size of the convolutional layer while preserving the most important features.
- **Fully Connected Layer**: Maps the pooled features to the output classes.
- **Dropout**: Used to regularize the network and prevent overfitting.

### **2.3 Key Hyperparameters**
- **vocab_size**: Size of the vocabulary used in the embedding layer.
- **embedding_dim**: Dimensionality of the word embeddings (e.g., 100, 300).
- **num_filters**: Number of filters to use in the convolutional layer (e.g., 100, 256).
- **filter_sizes**: The kernel sizes to use for capturing different n-grams (e.g., [3, 4, 5]).
- **dropout**: Dropout rate for regularization (e.g., 0.5).

### **2.4 Rationale**
CNNs are computationally efficient and capable of capturing local features (such as word combinations) that are important in text classification tasks. They are highly parallelizable, making them suitable for fast training on large datasets. The use of different filter sizes allows the model to capture varying lengths of n-grams, improving its robustness to sentence structure.

---

## **3. Hybrid Model (CNN + RNN)**

### **3.1 Overview**
For more complex tasks, combining CNNs and RNNs into a hybrid model can improve performance by leveraging the strengths of both architectures. CNNs can be used to extract local features from the input text, while RNNs can capture long-term dependencies across the sequence.

### **3.2 Model Architecture**
```plaintext
+------------------------------------------------+
|                Embedding Layer                 |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|           1D Convolutional Layer               |
|  (Captures local features through n-grams)     |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|              Max-Pooling Layer                 |
|   (Reduces dimensionality and retains features)|
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|           Recurrent Layers (RNN/LSTM/GRU)      |
| (Processes extracted features over sequence)   |
+------------------------------------------------+
                      |
                      v
+------------------------------------------------+
|                Fully Connected Layer           |
|     (For classification, with softmax output)  |
+------------------------------------------------+
```

### **3.3 Rationale**
This hybrid approach allows the model to first focus on extracting local features (through CNN) and then learn sequential dependencies in the text (through RNN). This is especially useful for long text sequences where local patterns matter but capturing temporal dependencies is still crucial.

---

## **4. Model Comparison**

| Architecture       | Strengths                                 | Weaknesses                                | Best Use Case                           |
|--------------------|-------------------------------------------|-------------------------------------------|-----------------------------------------|
| **Vanilla RNN**    | Simple, fast to train                     | Suffers from vanishing gradient           | Short text sequences                    |
| **LSTM/GRU**       | Handles long-term dependencies, avoids vanishing gradient | Computationally more expensive            | Long text sequences, such as articles   |
| **CNN**            | Efficient, captures local patterns        | Cannot capture long-term dependencies     | Short or medium-length texts with local patterns (e.g., n-grams) |
| **CNN + RNN**      | Leverages both local feature extraction and long-term dependencies | More complex, higher computational cost   | Complex, long sequences                 |

---

## **5. Model Configuration**
### **5.1 RNN Model Example (LSTM)**
```yaml
model:
  type: lstm
  vocab_size: 50000
  embedding_dim: 300
  hidden_dim: 256
  n_layers: 2
  dropout: 0.5
  bidirectional: true
  rnn_type: LSTM
```

### **5.2 CNN Model Example**
```yaml
model:
  type: cnn
  vocab_size: 50000
  embedding_dim: 300
  num_filters: 256
  filter_sizes: [3, 4, 5]
  dropout: 0.5
```

---

## **6. Future Enhancements**
1. **Transformer Models**: Implement **BERT**, **GPT**, or **Transformer**-based architectures for NLP tasks. These models have proven superior in many NLP benchmarks.
2. **Attention Mechanisms**: Introduce

 attention layers to focus on important parts of the sequence, improving RNN performance.
3. **Transfer Learning**: Use pre-trained embeddings like **GloVe** or **FastText** to initialize the embedding layer, which often results in better model performance for text classification tasks.
4. **Advanced Regularization**: Techniques like **Layer Normalization** or **Batch Normalization** can be added to the models to stabilize training and improve generalization.

---

## **7. Conclusion**
This project leverages multiple architectures—RNN, CNN, and a hybrid CNN + RNN—for NLP tasks such as text classification. Each model architecture is tailored to different types of text data, enabling the system to handle a wide range of scenarios. Future work includes expanding the architecture to more advanced models like Transformers.
