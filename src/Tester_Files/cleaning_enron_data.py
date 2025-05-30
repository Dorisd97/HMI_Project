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


def format_enron_email_body(text):
    if not isinstance(text, str):
        return text

    # Step 1: Normalize and clean up
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[<>]', '', text)
    text = text.replace('\\\\', '/').replace('\\', '/')
    text = re.sub(r'[-=_]{3,}', '', text)

    # Step 2: Break reply headers into blocks
    # Example: "Jeff Dasovich@EES on 04/14/2000 04:36:35 PM" becomes:
    # ----------------------------
    # Jeff Dasovich @ EES
    # 04/14/2000 04:36:35 PM
    # ----------------------------
    text = re.sub(
        r'(\n)?(\w[\w\s\.-]+@\w[\w\.-]+)\s+on\s+(\d{2}/\d{2}/\d{4}|\d{1,2} \w+ \d{4})\s+(\d{1,2}:\d{2}(?::\d{2})?\s?[APMapm]{2})',
        lambda m: f"\n\n----------------------------\n{m.group(2).replace('@', ' @ ')}\n{m.group(3)} {m.group(4)}\n----------------------------",
        text
    )

    # Step 3: Ensure "To:", "From:", etc. are on their own lines
    headers = ['To:', 'From:', 'Cc:', 'Bcc:', 'Subject:', 'Sent:', 'Date:', 'Forwarded by', 'Original Message']
    for kw in headers:
        text = re.sub(rf'(?<!\n)({re.escape(kw)})', r'\n\1', text)

    # Step 4: Merge broken lines intelligently
    lines = text.split('\n')
    result = []
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                result.append(' '.join(buffer).strip())
                buffer = []
        else:
            if buffer and not buffer[-1].endswith(('.', '?', '!', ':')) and not line[0].isupper():
                buffer[-1] += ' ' + line
            else:
                buffer.append(line)

    if buffer:
        result.append(' '.join(buffer).strip())

    # Step 5: Join all paragraphs with double line breaks
    return '\n\n'.join(result).strip()


def clean_text(text):
    return format_enron_email_body(text)


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

        # Format date field
        record["Date"] = format_date_to_european(record.get("Date", ""))

        # Clean address fields
        for key in ["From", "To", "X-cc", "X-bcc"]:
            if key in record:
                original = record[key]
                record[key] = clean_email_addresses(record[key])
                logger.debug(f"Cleaned {key}: {original} → {record[key]}")

        # Format the body properly
        if "Body" in record:
            original = record["Body"]
            formatted = format_enron_email_body(original)
            record["Body"] = formatted
            logger.info(f"Original Body (first 200 chars):\n{original[:200].strip()}...\n")
            logger.info(f"Formatted Body:\n{formatted[:500]}...\n")

        # Final pass for other fields
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