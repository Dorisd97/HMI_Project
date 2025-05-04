import os
import re
import pandas as pd
from src.config.config import UNZIP_DIR, REFINED_CSV_PATH
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

# Headers to extract
HEADER_KEYS = [
    "Message-ID", "Date", "From", "To", "Subject", "Mime-Version",
    "Content-Type", "Content-Transfer-Encoding", "X-From", "X-To",
    "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName"
]

def parse_email_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        headers = {}
        current_key = None
        body_lines = []
        in_headers = True

        for i, line in enumerate(lines):
            line = line.rstrip('\n')

            if in_headers:
                # Empty line marks end of headers
                if line.strip() == "":
                    in_headers = False
                    continue

                # Continuation of previous header
                if line.startswith((' ', '\t')) and current_key:
                    headers[current_key] += ' ' + line.strip()
                else:
                    match = re.match(r'^([\w\-]+):\s*(.*)', line)
                    if match:
                        current_key, value = match.groups()
                        headers[current_key] = value.strip()
                    else:
                        # Edge case: malformed header, treat as body
                        in_headers = False
                        body_lines.append(line)
            else:
                body_lines.append(line)

        record = {key: headers.get(key, "") for key in HEADER_KEYS}
        record["Body"] = "\n".join(body_lines).strip()
        record["SourceFile"] = file_path

        return record

    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return None

def extract_emails_to_csv(output_path):
    records = []

    logger.info(f"Scanning directory: {UNZIP_DIR}")
    for root, dirs, files in os.walk(UNZIP_DIR):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() == ".txt" or ext == "" or re.fullmatch(r"\.\s*\d+", ext):
                file_path = os.path.join(root, file)
                record = parse_email_file(file_path)
                if record:
                    records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Extracted {len(df)} emails and saved to '{output_path}'")

if __name__ == "__main__":
    extract_emails_to_csv(REFINED_CSV_PATH)