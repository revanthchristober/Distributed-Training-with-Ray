# download_data.py

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import kaggle
import logging
import logging.config
import yaml  # For loading YAML configuration
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
logger = logging.getLogger('my_logger')  # Custom logger defined in logging.yaml

# Paths and constants
RAW_DATA_DIR = Path('./data/raw/')
KAGGLE_DATASET = 'amananandrai/ag-news-classification-dataset'
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk size for large file downloads

# Ensure raw data directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Download from Kaggle using the Kaggle API
def download_kaggle_dataset(dataset_name: str, download_dir: Path):
    """Download a dataset from Kaggle using the Kaggle API."""
    logger.info(f'Downloading dataset {dataset_name} from Kaggle...')
    
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
        logger.info(f'Dataset downloaded and extracted to {download_dir}')
    except Exception as e:
        logger.error(f'Error downloading dataset from Kaggle: {e}')
        raise

# Download large datasets with chunked requests
def download_large_file(url: str, output_path: Path):
    """Download a large file in chunks with progress tracking."""
    logger.info(f'Starting download of {url}')
    
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as file, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    size = file.write(chunk)
                    bar.update(size)
        logger.info(f'File downloaded successfully: {output_path}')
    except Exception as e:
        logger.error(f'Error downloading large file: {e}')
        raise

# Extract compressed files (supports .zip and .tar)
def extract_compressed_file(file_path: Path, extract_to: Path):
    """Extract compressed files (supports .zip, .tar.gz, .tar.bz2)."""
    logger.info(f'Extracting {file_path} to {extract_to}')
    
    try:
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.suffix in ['.tar', '.gz', '.bz2']:
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.warning(f'Unsupported file type for extraction: {file_path}')
        
        logger.info(f'Extraction complete: {file_path}')
    except Exception as e:
        logger.error(f'Error extracting {file_path}: {e}')
        raise

# Download and extract a dataset from a URL (non-Kaggle)
def download_and_extract(url: str, download_dir: Path):
    """Download a dataset from a given URL and extract if compressed."""
    file_name = url.split('/')[-1]
    file_path = download_dir / file_name

    if not file_path.exists():
        download_large_file(url, file_path)

    # Extract if it's a compressed file
    if file_path.suffix in ['.zip', '.tar', '.gz', '.bz2']:
        extract_compressed_file(file_path, download_dir)

# Main function to orchestrate dataset downloads
def main():
    # Example: Download from Kaggle
    try:
        download_kaggle_dataset(KAGGLE_DATASET, RAW_DATA_DIR)
    except Exception as e:
        logger.error(f'Kaggle download failed: {e}')

    # Example: Download a large file from URL
    try:
        large_file_url = 'https://example.com/large-dataset.zip'  # Replace with actual dataset URL
        download_and_extract(large_file_url, RAW_DATA_DIR)
    except Exception as e:
        logger.error(f'URL download failed: {e}')

if __name__ == '__main__':
    main()
