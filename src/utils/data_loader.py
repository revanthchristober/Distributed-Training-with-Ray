# data_loader.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import spacy

# Initialize logger
import logging
logger = logging.getLogger('my_logger.utils.data_loader')

# Load Spacy tokenizer (English model)
nlp = spacy.load("en_core_web_sm")

class TextDataset(Dataset):
    """
    Custom Dataset class for text data.
    Args:
        data (pd.DataFrame): The data in DataFrame format with text and labels.
        tokenizer (spacy.lang): Spacy tokenizer for tokenizing the text.
        vocab_size (int): Maximum size of the vocabulary (number of unique tokens).
        max_seq_len (int): Maximum length of a sequence (padded/truncated to this length).
    """
    def __init__(self, data, tokenizer, vocab_size, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Build vocabulary from data
        self.word2idx = self.build_vocab(self.data['Title'] + " " + self.data['Description'])
        logger.info(f"Vocabulary built with {len(self.word2idx)} tokens.")

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     # Ensure the text is a writable string by making a copy
    #     text = str(self.data.iloc[idx]['Title']) + " " + str(self.data.iloc[idx]['Description'])
    #     text = text[:]  # Ensure it's a writable copy

    #     # Tokenize and convert to sequence of word indices
    #     tokens = self.tokenizer(text)
    #     token_ids = [self.word2idx.get(token.text, self.word2idx["<UNK>"]) for token in tokens]

    #     # Return padded token_ids and label
    #     token_ids = torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long)  # Truncate if necessary
    #     label = torch.tensor(self.data.iloc[idx]['Class Index'] - 1, dtype=torch.long)  # Class Index label
    #     return token_ids, label

    def __getitem__(self, idx):
        # Ensure the text is a writable string by concatenating Title and Description
        text = str(self.data.iloc[idx]['Title']) + " " + str(self.data.iloc[idx]['Description'])

        # Tokenize the text using the tokenizer
        tokens = self.tokenizer(text)

        # Convert tokens to a sequence of word indices
        token_ids = [self.word2idx.get(token.text, self.word2idx["<UNK>"]) for token in tokens]

        # Pad or truncate the token_ids to the maximum sequence length
        token_ids = torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long)

        # Convert the class index to a zero-based label tensor
        label = torch.tensor(self.data.iloc[idx]['Class Index'] - 1, dtype=torch.long)

        return token_ids, label

    def build_vocab(self, texts):
        """
        Build a word-to-index dictionary based on the tokenized text data.
        Args:
            texts (pd.Series): Series containing all the text data.
        Returns:
            word2idx (dict): Dictionary mapping tokens to unique indices.
        """
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for text in tqdm(texts, desc="Building Vocabulary"):
            tokens = self.tokenizer(text)
            for token in tokens:
                if token.text not in word2idx and len(word2idx) < self.vocab_size:
                    word2idx[token.text] = idx
                    idx += 1
        return word2idx

def pad_collate_fn(batch):
    """
    Collate function to apply padding for variable-length sequences in each batch.
    Args:
        batch (list): List of tuples (sequence, label) for the current batch.
    Returns:
        padded_sequences (torch.Tensor): Batch of padded token sequences.
        labels (torch.Tensor): Batch of corresponding labels.
        seq_lengths (torch.Tensor): Batch of sequence lengths before padding.
    """
    # Sort batch by sequence length (for RNN efficiency)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences, labels = zip(*batch)
    seq_lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return padded_sequences, labels, seq_lengths

def load_data(processed_data_dir, batch_size, vocab_size=10000, max_seq_len=500):
    """
    Load the preprocessed data and create DataLoader objects for training, validation, and testing.

    Args:
        processed_data_dir (str): Path to the directory where processed CSV files are stored.
        batch_size (int): Batch size for DataLoader.
        vocab_size (int): Maximum vocabulary size (for tokenization).
        max_seq_len (int): Maximum sequence length (for padding/truncation).

    Returns:
        train_loader (DataLoader): DataLoader object for training set.
        val_loader (DataLoader): DataLoader object for validation set.
        test_loader (DataLoader): DataLoader object for test set.
    """
    logger.info(f"Loading data from {processed_data_dir}...")

    # Load the preprocessed data from CSV files
    train_data = pd.read_csv(Path(processed_data_dir) / 'train_data.csv')
    val_data = pd.read_csv(Path(processed_data_dir) / 'val_data.csv')
    test_data = pd.read_csv(Path(processed_data_dir) / 'test_data.csv')

    # Create Dataset objects
    train_dataset = TextDataset(train_data, nlp, vocab_size, max_seq_len)
    val_dataset = TextDataset(val_data, nlp, vocab_size, max_seq_len)
    test_dataset = TextDataset(test_data, nlp, vocab_size, max_seq_len)

    # Create DataLoader objects with the custom collate function for padding
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    logger.info(f"Data loaded successfully with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    
    return train_loader, val_loader, test_loader
