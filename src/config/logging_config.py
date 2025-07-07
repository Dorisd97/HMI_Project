import logging
import sys

def setup_logger(name='dataset_logger'):
    # Create or get a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to capture all levels (DEBUG and above)

    # Clear existing handlers to avoid duplicate logs if this function is called multiple times
    logger.handlers.clear()

    # Define the log message format: timestamp, log level, and message
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Create a console handler that outputs logs to standard output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on the console
    console_handler.setFormatter(formatter)  # Apply the formatter to the console handler
    logger.addHandler(console_handler)  # Attach the handler to the logger

    return logger  # Return the configured logger instance