# rnn_model.py

import torch
import torch.nn as nn
import logging

# Initialize logger for the RNN model module
logger = logging.getLogger('distributed_training.models.rnn_model')

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional=True, dropout=0.5, rnn_type="lstm"):
        """
        RNN-based model for text classification.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embedding_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): Dimensionality of RNN hidden states.
            output_dim (int): Number of output classes.
            n_layers (int): Number of recurrent layers.
            bidirectional (bool): Whether to use bidirectional RNN/LSTM/GRU.
            dropout (float): Dropout rate.
            rnn_type (str): Type of RNN (Options: "lstm", "gru").
        """
        super(RNNModel, self).__init__()

        logger.info(f"Initializing RNN Model with {rnn_type.upper()}...")

        # Embedding Layer (Convert words to dense vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        logger.info(f"Embedding Layer initialized with vocab_size={vocab_size}, embedding_dim={embedding_dim}.")

        # RNN Layer (You can choose between LSTM or GRU)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                               bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            logger.error("Invalid RNN type specified! Choose 'lstm' or 'gru'.")
            raise ValueError("Unsupported RNN type. Choose 'lstm' or 'gru'.")
        
        logger.info(f"{rnn_type.upper()} Layer initialized with hidden_dim={hidden_dim}, n_layers={n_layers}, bidirectional={bidirectional}.")

        # Fully Connected Layer (For outputting class probabilities)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        logger.info(f"Fully Connected Layer initialized with output_dim={output_dim}.")

        # Dropout Layer (To prevent overfitting)
        self.dropout = nn.Dropout(dropout)

        # Softmax for output class probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, text_lengths):
        """
        Forward pass for RNNModel.

        Args:
            text (torch.Tensor): Input tensor of token indices [batch_size, seq_len].
            text_lengths (torch.Tensor): Lengths of the input sequences.

        Returns:
            output (torch.Tensor): Logits for each class [batch_size, output_dim].
        """
        logger.debug("Starting forward pass of RNN model.")
        
        # Embedding Layer (convert word indices to embeddings)
        embedded = self.dropout(self.embedding(text))
        logger.debug("Completed embedding layer.")

        # Pack padded batch for the RNN
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN Forward Pass
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        logger.debug("Completed RNN forward pass.")

        # Unpack the output (sequence, batch_size, hidden_size)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # If using bidirectional RNN, concatenate the hidden states from both directions
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Last hidden states from both directions
            logger.debug("Using bidirectional RNN, concatenated forward and backward hidden states.")
        else:
            hidden = hidden[-1]  # Only last hidden state for single-directional RNN
            logger.debug("Using single-directional RNN.")

        # Fully Connected Layer
        hidden = self.dropout(hidden)  # Apply dropout before the final layer
        output = self.fc(hidden)
        logger.debug("Completed forward pass through Fully Connected layer.")
        
        return self.softmax(output)