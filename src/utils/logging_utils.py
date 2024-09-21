import logging
import logging.config
import os
import sys
import json
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from pythonjsonlogger import jsonlogger
import yaml
import datetime

class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that extends the pythonjsonlogger.JsonFormatter to add extra fields to the log.
    """
    def add_fields(self, log_record, record, message_dict):
        super(CustomJSONFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['filename'] = record.pathname
        log_record['lineno'] = record.lineno
        log_record['funcName'] = record.funcName
        log_record['message'] = record.getMessage()


def setup_logger(
        name=None, 
        log_level=logging.INFO, 
        log_to_console=True, 
        log_to_file=True, 
        log_file_path='logs/app.log',
        json_format=False,
        max_log_size=10*1024*1024,  # 10 MB
        backup_count=5,
        rotation_interval=None,
        colored_logs=False
    ):
    """
    Sets up a logger with options for console and file logging, JSON formatting, and rotation.

    Parameters:
    - name (str): Name of the logger.
    - log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    - log_to_console (bool): Whether to log to console.
    - log_to_file (bool): Whether to log to file.
    - log_file_path (str): File path for the log file.
    - json_format (bool): Whether to use JSON formatting for the logs.
    - max_log_size (int): Maximum size for log rotation (if no time-based rotation).
    - backup_count (int): Number of backup logs to keep.
    - rotation_interval (str): Time interval for rotating logs (e.g., 'midnight', 'H', 'D').
    - colored_logs (bool): Whether to use colored logs in the console.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    log_handlers = []

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter() if colored_logs else logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        log_handlers.append(console_handler)

    # File Handler (with optional rotation)
    if log_to_file:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        if rotation_interval:  # Time-based rotation
            file_handler = TimedRotatingFileHandler(log_file_path, when=rotation_interval, backupCount=backup_count)
        else:  # Size-based rotation
            file_handler = RotatingFileHandler(log_file_path, maxBytes=max_log_size, backupCount=backup_count)

        if json_format:
            file_handler.setFormatter(CustomJSONFormatter('%(timestamp)s %(level)s %(filename)s %(lineno)d %(message)s'))
        else:
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        log_handlers.append(file_handler)

    # Add handlers to logger
    for handler in log_handlers:
        handler.setLevel(log_level)
        logger.addHandler(handler)

    return logger


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter for colored logs in the console.
    """
    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m' # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            color = self.COLORS[levelname]
            record.msg = f"{color}{record.msg}{self.RESET}"
        return super(ColoredFormatter, self).format(record)


def load_logging_config(config_file: str):
    """
    Load logging configuration from a YAML file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - None
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Logging configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def log_request_info(logger, request):
    """
    Log detailed information about a web request.

    Parameters:
    - logger (logging.Logger): Logger instance to use for logging.
    - request (object): Web request object (e.g., Flask or FastAPI request).

    Returns:
    - None
    """
    logger.info(f"Request path: {request.path}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request query params: {request.query_params}")


def log_model_training_progress(logger, epoch, loss, accuracy):
    """
    Log model training progress during each epoch.

    Parameters:
    - logger (logging.Logger): Logger instance.
    - epoch (int): Current epoch number.
    - loss (float): Training loss.
    - accuracy (float): Training accuracy.

    Returns:
    - None
    """
    logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")


def log_inference(logger, input_data, prediction):
    """
    Log inference input and output for a machine learning model.

    Parameters:
    - logger (logging.Logger): Logger instance.
    - input_data (any): Input data for inference.
    - prediction (any): Model's prediction.

    Returns:
    - None
    """
    logger.info(f"Inference input: {input_data}")
    logger.info(f"Inference output (prediction): {prediction}")


def create_log_directory(log_dir):
    """
    Create a directory for storing log files if it does not already exist.

    Parameters:
    - log_dir (str): Path to the log directory.

    Returns:
    - None
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Log directory created at {log_dir}")


# Example usage of the logging utility
if __name__ == "__main__":
    # Set up structured logging with file rotation
    logger = setup_logger(
        name='advanced_logger',
        log_level=logging.DEBUG,
        log_to_console=True,
        log_to_file=True,
        log_file_path='logs/app.log',
        json_format=True,  # Set to False for regular text logs
        max_log_size=5 * 1024 * 1024,  # 5 MB file rotation
        backup_count=3,
        rotation_interval=None,  # No time-based rotation
        colored_logs=True
    )

    logger.info("This is an info log")
    logger.debug("This is a debug log with detailed information")
    logger.error("This is an error log with some issue description")

    # Simulate logging a web request
    class MockRequest:
        def __init__(self):
            self.path = "/api/v1/resource"
            self.method = "GET"
            self.headers = {"User-Agent": "test-client"}
            self.query_params = {"id": "123"}
    
    mock_request = MockRequest()
    log_request_info(logger, mock_request)

    # Simulate logging during model training
    log_model_training_progress(logger, epoch=5, loss=0.2345, accuracy=92.5)

    # Simulate inference logging
    input_data = {"input": [1, 2, 3, 4]}
    prediction = {"prediction": "class_1"}
    log_inference(logger, input_data, prediction)
