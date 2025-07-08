import os
import re
from src.config.config import UNZIP_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

def rename_files_to_txt():
    for root, dirs, files in os.walk(UNZIP_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)

            # If file has no extension or weird numeric extension, rename it
            if ext == "" or re.fullmatch(r"\.\s*\d+", ext):
                new_filename = filename + ".txt"
                new_path = os.path.join(root, new_filename)

                try:
                    os.rename(file_path, new_path)
                    logger.info(f"Renamed '{filename}' to '{new_filename}'")
                except Exception as e:
                    logger.error(f"Failed to rename '{file_path}': {e}")

if __name__ == "__main__":
    rename_files_to_txt()
