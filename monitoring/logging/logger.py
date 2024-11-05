import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Define the directory for logs
log_directory = "/var/log/anomaly_detection"
os.makedirs(log_directory, exist_ok=True)

# Logger configuration
logger = logging.getLogger("AnomalyDetectionLogger")
logger.setLevel(logging.DEBUG)

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler for rotating logs daily and keeping 7 backups
file_handler = TimedRotatingFileHandler(
    os.path.join(log_directory, "anomaly_detection.log"),
    when="midnight",
    interval=1,
    backupCount=7
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Usage
def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)

def log_debug(message):
    logger.debug(message)

def log_warning(message):
    logger.warning(message)

log_info("Logger initialized successfully")
log_error("An error occurred in the anomaly detection model")