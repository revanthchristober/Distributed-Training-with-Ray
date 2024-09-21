import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import spacy
import re
import logging
import logging.config
import yaml  # To load YAML configuration
from datetime import datetime

# Load logging configuration from YAML
CONFIG_PATH = Path('./config/logging.yaml')

def setup_logging(config_path=CONFIG_PATH):
    """Set up logging configuration from a YAML file."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        print(f"Logging configuration file {config_path} not found. Using default logging setup.")

# Initialize logging
setup_logging()
logger = logging.getLogger('distributed_training_preprocessing_logger')  # Using the custom logger defined in the YAML

# Paths and tokenizer settings
RAW_DATA_DIR = Path('./data/raw/')
PROCESSED_DATA_DIR = Path('./data/processed/')
DATA_FILE_NAME = 'train.csv'
TOKENIZER_MODEL = 'en_core_web_sm'  # Spacy tokenizer

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Preprocess AG News dataset (clean, tokenize, normalize, and transform) with progress bars
def preprocess_data(file_path: Path, balance_data=False, chunk_size=10000):
    logger.info(f'Loading and preprocessing data from {file_path}...')
    
    try:
        # Load dataset in chunks to handle large files efficiently
        logger.info(f'Reading dataset in chunks (size={chunk_size})...')
        chunks = pd.read_csv(file_path, chunksize=chunk_size)

        processed_data = []
        for chunk in chunks:
            logger.info(f'Processing chunk with {len(chunk)} records...')

            # Drop duplicates and handle missing values
            chunk = chunk.drop_duplicates()
            chunk = chunk.dropna(subset=['Title', 'Description'])
            logger.info(f'Data cleaned: removed duplicates and missing values.')

            # Shuffle the dataset
            chunk = shuffle(chunk, random_state=42)
            logger.info(f'Dataset loaded and shuffled successfully.')

            # Clean and normalize text data with progress bars
            logger.info('Cleaning, normalizing, and tokenizing text data...')
            
            # Combine 'Title' and 'Description' for the full text
            chunk['text'] = chunk['Title'] + ' ' + chunk['Description']

            # Apply advanced text cleaning with tqdm progress bar
            chunk['text'] = [clean_text(text) for text in tqdm(chunk['text'], desc="Cleaning and Normalizing Text")]

            # Load Spacy tokenizer
            nlp = spacy.load(TOKENIZER_MODEL)

            # Tokenize and lemmatize text with tqdm progress bar
            chunk['tokens'] = [tokenize_text(text, nlp) for text in tqdm(chunk['text'], desc="Tokenizing and Lemmatizing Text")]

            processed_data.append(chunk)

        # Concatenate all processed chunks
        df = pd.concat(processed_data)

        # Optional: Handle class imbalance by upsampling/downsampling (if specified)
        if balance_data:
            logger.info('Balancing dataset for class distribution...')
            df = balance_dataset(df)

        # Split dataset into train, validation, and test sets (80% train, 10% val, 10% test)
        train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Save processed data
        save_dataset(train_data, 'train')
        save_dataset(val_data, 'val')
        save_dataset(test_data, 'test')

        logger.info(f'Preprocessing complete. Processed data saved to {PROCESSED_DATA_DIR}')
    except Exception as e:
        logger.error(f'Error during preprocessing: {e}')
        raise

# Clean and normalize the text data (advanced cleaning)
def clean_text(text):
    """Clean and normalize the text data by removing unnecessary characters and URLs."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Tokenize and lemmatize the text using Spacy
def tokenize_text(text, nlp):
    """Tokenize and lemmatize the text using Spacy while removing stop words."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]  # Lemmatization and removing stop words
    return tokens

# Handle imbalanced data (optional)
def balance_dataset(df):
    """Balance the dataset by either upsampling or downsampling based on class distribution."""
    label_counts = df['Class Index'].value_counts()
    max_count = label_counts.max()

    # Upsample minority classes
    balanced_df = pd.DataFrame()
    for label in label_counts.index:
        label_df = df[df['Class Index'] == label]
        if len(label_df) < max_count:
            label_df = resample(label_df, replace=True, n_samples=max_count, random_state=42)
        balanced_df = pd.concat([balanced_df, label_df])

    logger.info(f'Dataset balanced to class sizes: {label_counts.to_dict()}')
    return balanced_df

# Save the dataset as CSV files
def save_dataset(dataframe, dataset_type):
    """Save processed data to a CSV file."""
    output_file = PROCESSED_DATA_DIR / f'{dataset_type}_data.csv'
    try:
        dataframe.to_csv(output_file, index=False)
        logger.info(f'Saved {dataset_type} data to {output_file}')
    except Exception as e:
        logger.error(f'Error saving {dataset_type} data: {e}')
        raise

# Main function to preprocess dataset
def main():
    dataset_path = RAW_DATA_DIR / DATA_FILE_NAME
    preprocess_data(dataset_path, balance_data=False)

if __name__ == '__main__':
    main()