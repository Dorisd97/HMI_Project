import os
import re
import pandas as pd
import natsort
from src.config.config import UNZIP_DIR, REFINED_CSV_PATH
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
            raise ValueError(f"Could not determine body start in {file_path}")

        body_lines = lines[body_start_index:]
        body_text = "\n".join(body_lines).strip()
        if not body_text:
            raise ValueError(f"Empty email body in {file_path}")

        from_blocks = [m.start() for m in re.finditer(r'^From: .*', body_text, re.MULTILINE)]
        reply_chains = []
        if from_blocks:
            from_blocks.append(len(body_text))
            for i in range(1, len(from_blocks)):
                start = from_blocks[i - 1]
                end = from_blocks[i]
                reply_chains.append(body_text[start:end].strip())
            main_body = body_text[:from_blocks[0]].strip()
        else:
            main_body = body_text

        record = {key: headers.get(key, "") for key in HEADER_KEYS}
        record["Body"] = main_body.replace('"""', '""')
        record["SourceFile"] = os.path.basename(file_path)  # ✔️ Use filename only

        for idx, reply in enumerate(reply_chains):
            record[f"ReplyChain{idx + 1}"] = reply.replace('"""', '""')

        if not record["SourceFile"].strip():
            raise ValueError(f"SourceFile is empty in {file_path}")

        return record

    except Exception as e:
        logger.error(f"Error parsing email in {file_path}: {e}")
        raise

def extract_emails_to_csv(output_path, batch_size=50):
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
    batch_files = [file_map[name] for name in sorted_file_names[:batch_size]]

    existing_df = pd.DataFrame()
    processed_filenames = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, dtype=str)
            processed_filenames = set(
                os.path.basename(name) for name in existing_df["SourceFile"].dropna().unique()
            )
            logger.info(f"Loaded {len(existing_df)} existing rows from CSV")
        except Exception as e:
            logger.error(f"Error reading existing CSV: {e}")
            return

    # ✔️ Filter out files based on just the filename
    batch_files = [fp for fp in batch_files if os.path.basename(fp) not in processed_filenames]
    logger.info(f"{len(batch_files)} new files to process after deduplication")

    all_records = []
    for file_path in batch_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                record = parse_single_email(content, file_path)
                all_records.append(record)
                logger.info(f"Parsed: {file_path}")
        except Exception as e:
            logger.error(f"Critical error in {file_path}: {e}")
            raise RuntimeError(f"Aborting due to error in {file_path}")

    if not all_records:
        logger.info("No new records to add. CSV is up to date.")
        return

    new_df = pd.DataFrame(all_records)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).fillna("")

    try:
        combined_df.to_csv(output_path, index=False, quoting=1, escapechar='\\', lineterminator='\n')
        logger.info(f"Updated CSV with total {len(combined_df)} records at '{output_path}'")
    except Exception as e:
        logger.error(f"Error writing updated CSV: {e}")

if __name__ == "__main__":
    extract_emails_to_csv(REFINED_CSV_PATH, batch_size=100)
