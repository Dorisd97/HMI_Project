import os
import re
import json
import pandas as pd
import natsort
from src.config.config import UNZIP_DIR, REFINED_JSON_PATH  # This is a full path like './data/enron_output.json'
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

HEADER_KEYS = [
    "Message-ID", "Date", "From", "To", "Subject", "Mime-Version",
    "Content-Type", "Content-Transfer-Encoding", "X-From", "X-To",
    "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName"
]

def parse_single_email(email_text, file_path):
    try:
        headers = {}
        current_key = None
        lines = email_text.splitlines()
        body_start_index = None

        for i, line in enumerate(lines):
            if line.startswith((' ', '\t')) and current_key:
                headers[current_key] += ' ' + line.strip()
            else:
                match = re.match(r'^([\w\-]+):\s*(.*)', line)
                if match:
                    current_key, value = match.groups()
                    headers[current_key] = value.strip()
                    if current_key == "X-FileName":
                        body_start_index = i + 1
                    elif current_key == "X-Origin" and body_start_index is None:
                        body_start_index = i + 1

        if body_start_index is None or body_start_index >= len(lines):
            logger.warning(f"No body found in: {file_path}")
            body_text = ""
        else:
            body_lines = lines[body_start_index:]
            body_text = "\n".join(body_lines).strip()

        # Detect and extract structured forwarded messages
        body_chain = []
        message_blocks = re.split(r"(?=^To: )", body_text, flags=re.MULTILINE)

        for block in message_blocks:
            if not block.strip():
                continue
            entry = {}
            lines = block.strip().splitlines()
            body_lines = []
            in_body = False

            for line in lines:
                if not in_body:
                    if re.match(r"^To:\s*(.*)", line):
                        entry["To"] = line[3:].strip()
                    elif re.match(r"^cc:\s*(.*)", line, re.IGNORECASE):
                        entry["cc"] = line[3:].strip()
                    elif re.match(r"^Subject:\s*(.*)", line, re.IGNORECASE):
                        entry["Subject"] = line[8:].strip()
                        in_body = True  # start body after subject
                else:
                    body_lines.append(line)

            if body_lines:
                entry["Body"] = "\n".join(body_lines).strip()

            if entry:
                body_chain.append(entry)

        # Use the first block as main body if no 'To:' found
        if not body_chain:
            main_body = body_text.strip()
        else:
            main_body = message_blocks[0].strip()

        # Compile the full record
        record = {key: headers.get(key, "") for key in HEADER_KEYS}
        record["Body"] = main_body.replace('"""', '""')
        record["SourceFile"] = os.path.basename(file_path)

        if body_chain:
            record["BodyChain"] = body_chain

        return record

    except Exception as e:
        logger.error(f"Error parsing email in {file_path}: {e}")
        raise

def generate_unique_json_path(base_path):
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def extract_emails_to_json(base_output_path, batch_size=0):
    output_path = generate_unique_json_path(base_output_path)

    all_files = []
    file_map = {}

    for root, dirs, files in os.walk(UNZIP_DIR):
        for file in sorted(files):
            _, ext = os.path.splitext(file)
            if ext.lower() == ".txt" or ext == "" or re.fullmatch(r"\.\s*\d+", ext):
                file_path = os.path.join(root, file)
                file_map[file] = file_path
                all_files.append(file_path)

    total = len(all_files)
    logger.info(f"Total eligible files found: {total}")

    sorted_file_names = natsort.natsorted(file_map.keys())
    batch_files = (
        [file_map[name] for name in sorted_file_names[:batch_size]]
        if batch_size > 0 else
        [file_map[name] for name in sorted_file_names]
    )

    logger.info(f"Processing {len(batch_files)} files. Output: {os.path.basename(output_path)}")

    new_records = []
    for file_path in batch_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                record = parse_single_email(content, file_path)
                new_records.append(record)
                logger.info(f"Parsed: {file_path}")
        except Exception as e:
            logger.error(f"Critical error in {file_path}: {e}")
            raise RuntimeError(f"Aborting due to error in {file_path}")

    if not new_records:
        logger.info("No records found. No JSON created.")
        return

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_records, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(new_records)} records to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Error writing new JSON file: {e}")


if __name__ == "__main__":
    extract_emails_to_json(REFINED_JSON_PATH, batch_size=0)