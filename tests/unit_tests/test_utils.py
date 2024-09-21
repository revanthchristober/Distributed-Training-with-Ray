import pytest
from unittest.mock import patch
from src.utils.data_loader import load_data
from src.utils.logging_utils import setup_logger

@pytest.fixture
def mock_data():
    """
    Fixture to provide mock data loading configuration.
    """
    return "/path/to/data", 32  # Mock processed data directory and batch size

@patch('src.utils.data_loader.pd.read_csv')
def test_data_loading(mock_read_csv, mock_data):
    """
    Test the data loading function.
    """
    # Mock return value from pandas read_csv
    mock_read_csv.return_value = {
        "Title": ["Sample title"] * 100,
        "Description": ["Sample description"] * 100,
        "Class Index": [1] * 100
    }
    
    train_loader, val_loader, test_loader = load_data(*mock_data)
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

def test_logger_setup():
    """
    Test logger setup to ensure correct configuration.
    """
    logger = setup_logger('test_logger', '/tmp/test.log')
    assert logger.name == 'test_logger', "Logger name should be 'test_logger'."
    assert logger.handlers, "Logger should have handlers set up."
