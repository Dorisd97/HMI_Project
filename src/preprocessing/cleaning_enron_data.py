import json
import re
import logging
from datetime import datetime
from src.config.config import REFINED_JSON_PATH, CLEANED_JSON_PATH_1

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_email_addresses(text):
    if not text:
        return text
    logger.debug("Cleaning email addresses.")
    parts = re.split(r'[;,]', text)
    cleaned = []
    for part in parts:
        part = part.strip()
        part = re.sub(r'<\s*([^<>]+?)\s*>', r'\1', part)
        cleaned.append(part)
    return ', '.join(cleaned)

def format_date_to_european(date_str):
    if not date_str:
        return date_str
    logger.debug(f"Formatting date: {date_str}")
    cleaned = re.sub(r'\s+\(.*?\)$', '', date_str.strip())
    cleaned = re.sub(r'\s+[+-]\d{4}$', '', cleaned)
    match = re.match(r'.*?(\d{1,2} \w{3} \d{4}) (\d{2}:\d{2}:\d{2})', cleaned)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        try:
            dt = datetime.strptime(date_part, "%d %b %Y")
            formatted = f"{dt.strftime('%d.%m.%Y')} {time_part}"
            logger.debug(f"Formatted to: {formatted}")
            return formatted
        except ValueError:
            logger.warning(f"Date parse error: {date_str}")
            return date_str
    return date_str

def clean_text(text):
    if not isinstance(text, str):
        return text

    original = text  # For debugging if needed

    # Remove angle-bracketed content (e.g., <email>, <metadata>)
    text = re.sub(r'<[^<>]+>', '', text)

    # Replace double slashes or paths
    text = text.replace('\\\\', '/').replace('\\', '/')

    # Remove horizontal lines or separators (---, ===, ___)
    text = re.sub(r'[-=_]{3,}', ' ', text)

    # Remove non-ASCII characters (emojis, symbols)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove excessive punctuation (e.g., .., --, !!, etc.)
    text = re.sub(r'[\.\-]{2,}', '.', text)
    text = re.sub(r'[\!\?]{2,}', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,?!:;])', r'\1', text)

    return text.strip()

def clean_record(obj):
    if isinstance(obj, dict):
        return {k: clean_record(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_record(item) for item in obj]
    elif isinstance(obj, str):
        return clean_text(obj)
    return obj

def process_first_n_emails(input_path, output_path, limit=50):
    logger.info(f"Loading JSON data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Processing first {limit} email records.")
    cleaned_batch = []

    for idx, record in enumerate(data[:limit]):
        logger.info(f"Processing record {idx + 1}/{limit}")

        # Clean date
        original_date = record.get("Date", "")
        record["Date"] = format_date_to_european(original_date)
        logger.debug(f"Cleaned Date: {record['Date']}")

        # Clean email fields
        for key in ["From", "To", "X-cc", "X-bcc"]:
            if key in record:
                original = record[key]
                record[key] = clean_email_addresses(record[key])
                logger.debug(f"Cleaned {key}: {original} → {record[key]}")

        # Deep clean entire record
        cleaned_record = clean_record(record)
        cleaned_batch.append(cleaned_record)

    logger.info(f"Saving cleaned data to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_batch, f, indent=2)
    logger.info("Cleaning process completed successfully.")

if __name__ == "__main__":
    process_first_n_emails(
        input_path=REFINED_JSON_PATH,
        output_path=CLEANED_JSON_PATH_1,
        limit=50
    )
