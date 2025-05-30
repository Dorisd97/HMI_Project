import os
import hashlib
import logging
from src.config.config import LOG_FILE_PATH
from src.preprocessing.Converting_File_To_JSON.dataset_setup import unzip_once

# Setup logging to 'log/duplicates_process_log.txt' under project root
LOGGING_PATH = os.path.join(os.path.dirname(LOG_FILE_PATH), 'duplicates_process_log.txt')
os.makedirs(os.path.dirname(LOGGING_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOGGING_PATH,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Step 1: Ensure dataset is unzipped
UNZIP_DIR = unzip_once()

# Step 2: Read all .txt files and compute their hashes
file_hashes = {}
for root, dirs, files in os.walk(UNZIP_DIR):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
                file_hash = hashlib.md5(file_content.encode('utf-8')).hexdigest()
                file_hashes.setdefault(file_hash, []).append(file_path)

# Step 3: Find and log duplicates
duplicate_files = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
logger.info(f"Total sets of duplicate files found: {len(duplicate_files)}")
for hash_value, files in duplicate_files.items():
    logger.info(f"Duplicate set (hash: {hash_value}):")
    for file in files:
        logger.info(f"  {file}")

# Step 4: Remove duplicates and log
with open(LOG_FILE_PATH, 'w') as log_file:
    for files in duplicate_files.values():
        for duplicate_file in files[1:]:
            os.remove(duplicate_file)
            log_msg = f"Deleted duplicate file: {duplicate_file}"
            log_file.write(log_msg + '\n')
            logger.info(log_msg)

logger.info(f"All deleted duplicates have been logged to {LOG_FILE_PATH}")
