import os
import zipfile
import logging
from src.config.config import ZIP_PATH, UNZIP_DIR

# Minimal logger setup: logs to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def unzip_once():
    """
    Unzips the Enron dataset if it hasn't been unzipped yet.
    Returns the path to the unzipped folder.
    """
    logger.info(f"ZIP_PATH: {ZIP_PATH}")
    logger.info(f"ZIP file exists? {os.path.exists(ZIP_PATH)}")
    logger.info(f"UNZIP_DIR: {UNZIP_DIR}")
    logger.info(f"UNZIP_DIR exists? {os.path.exists(UNZIP_DIR)}")

    if os.path.exists(UNZIP_DIR):
        logger.info(f"Using existing unzipped data at: {UNZIP_DIR}")
    else:
        logger.info(f"Unzipping data from: {ZIP_PATH}")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(UNZIP_DIR)
            logger.info(f"Data unzipped to: {UNZIP_DIR}")
        except Exception as e:
            logger.error(f"Failed to unzip data: {e}")
            raise

    return UNZIP_DIR

if __name__ == "__main__":
    unzip_once()
