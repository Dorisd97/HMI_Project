import os  # For file and directory operations
import re  # For regular expressions
import json  # For reading and writing JSON
import natsort  # For natural sorting of filenames
from src.config.config import UNZIP_DIR, REFINED_JSON_PATH  # Import config paths
from src.config.logging_config import setup_logger  # Import logger setup

logger = setup_logger(__name__)  # Initialize logger

HEADER_KEYS = [
    "Message-ID", "Date", "From", "To", "Subject", "Mime-Version",
    "Content-Type", "Content-Transfer-Encoding", "X-From", "X-To",
    "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName"
]  # List of email header fields to extract

def parse_single_email(content, file_path):
    try:
        headers = {}  # Dict to store headers
        lines = content.splitlines()  # Split content into lines
        current_key = None  # Track current header key
        body_start = 0  # Index where body starts

        # Parse headers with multiline support
        for i, line in enumerate(lines):
            if line.strip() == "":
                body_start = i + 1  # Empty line signals end of headers
                break
            if line.startswith((' ', '\t')) and current_key:
                headers[current_key] += ' ' + line.strip()  # Multiline header continuation
            else:
                match = re.match(r'^([\w\-]+):\s*(.*)', line)
                if match:
                    current_key, value = match.groups()
                    headers[current_key] = value.strip()

        # Extract body
        body = "\n".join(lines[body_start:]).strip()

        # Compile record
        record = {key: headers.get(key, "") for key in HEADER_KEYS}
        record["Body"] = body
        record["SourceFile"] = os.path.basename(file_path)

        return record

    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None

def extract_emails_to_json(base_output_path):
    file_map = {}  # Map of filename to full path

    # Collect eligible .txt or extension-less files
    for root, _, files in os.walk(UNZIP_DIR):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() == ".txt" or ext == "" or re.fullmatch(r"\.\s*\d+", ext):
                file_map[file] = os.path.join(root, file)

    sorted_file_names = natsort.natsorted(file_map.keys())  # Natural sort filenames
    sorted_file_paths = [file_map[name] for name in sorted_file_names]

    logger.info(f"Found {len(sorted_file_paths)} files for processing")

    records = []
    for file_path in sorted_file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                parsed = parse_single_email(content, file_path)
                if parsed:
                    records.append(parsed)
                    logger.info(f"Parsed: {file_path}")
        except Exception as e:
            logger.warning(f"Skipped {file_path} due to: {e}")

    try:
        os.makedirs(os.path.dirname(base_output_path), exist_ok=True)  # Ensure output directory exists
        with open(base_output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)  # Write all records to JSON
        logger.info(f"✅ Saved {len(records)} emails to: {base_output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save output JSON: {e}")

if __name__ == "__main__":
    extract_emails_to_json(REFINED_JSON_PATH)  # Run extraction if script is executed
