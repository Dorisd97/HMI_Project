import json
import re
from datetime import datetime
from src.config.config import REFINED_JSON_PATH, CLEANED_JSON_PATH

def clean_email_addresses(text):
    if not text:
        return text

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

    # Remove timezone info in parentheses and trailing offset
    cleaned = re.sub(r'\s+\(.*?\)$', '', date_str.strip())
    cleaned = re.sub(r'\s+[+-]\d{4}$', '', cleaned)

    # Match and extract date + time
    match = re.match(r'.*?(\d{1,2} \w{3} \d{4}) (\d{2}:\d{2}:\d{2})', cleaned)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        try:
            dt = datetime.strptime(date_part, "%d %b %Y")
            return f"{dt.strftime('%d.%m.%Y')} {time_part}"
        except ValueError:
            return date_str

    return date_str

def clean_enron_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        record["Date"] = format_date_to_european(record.get("Date", ""))
        for key in ["From", "To", "X-cc", "X-bcc"]:
            if key in record:
                record[key] = clean_email_addresses(record[key])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    clean_enron_json(REFINED_JSON_PATH, CLEANED_JSON_PATH)
