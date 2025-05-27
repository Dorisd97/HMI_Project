import json
import re
import logging
from src.config.config import CLEANED_JSON_PATH_1, BODY_CHAIN_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:", "sent:"]
ORDERED_KEYS = ["From", "Sent", "To", "Cc", "Bcc", "Subject", "Body"]


def capture_multiline_header_value(lines, start_idx, header_keys):
    """
    Capture multiline header value from start_idx until next header or
    an empty line followed by a non-header line (body start).
    """
    captured_lines = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        stripped_lower = line.lstrip().lower()

        # Check if line starts with header key
        if any(stripped_lower.startswith(h) for h in header_keys) or "forwarded by" in stripped_lower:
            break

        # Check for empty line followed by non-header line (body start)
        if line.strip() == "":
            # Peek next line if exists
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_line_lower = next_line.lstrip().lower()
                if not any(next_line_lower.startswith(h) for h in header_keys):
                    # Empty line followed by non-header means body start, stop
                    break

        captured_lines.append(line.rstrip())
        i += 1
    return " ".join(captured_lines).strip(), i


def extract_inline_headers(line, current_fields):
    # This still helps if multiple headers appear inline on a From line
    header_patterns = {
        "From": r'From:\s*(.*?)(?=\s+\w+:|$)',
        "Sent": r'Sent:\s*(.*?)(?=\s+\w+:|$)',
        "To": r'To:\s*(.*?)(?=\s+\w+:|$)',
        "Cc": r'Cc:\s*(.*?)(?=\s+\w+:|$)',
        "Bcc": r'Bcc:\s*(.*?)(?=\s+\w+:|$)',
        "Subject": r'Subject:\s*(.*)',  # only one line
    }
    for key, pattern in header_patterns.items():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            current_fields[key] = match.group(1).strip()
    return current_fields

def parse_body_chain_blocks(body_text, source_file=None):
    seen_from = False
    seen_to = False
    lines = body_text.splitlines()
    body_chain = []
    regular_body_lines = []

    current_fields = {key: "" for key in ORDERED_KEYS}
    current_body_lines = []
    in_chain = False
    block_started = False
    seen_from = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lower_line = stripped.lower()

        # 1. Detect "From:" header line — finalize previous block, start new block
        if re.match(r"^>*\s*from:", stripped, re.IGNORECASE):
            if block_started:
                current_fields["Body"] = "\n".join(current_body_lines).strip()
                body_chain.append({k: current_fields.get(k, "") for k in ORDERED_KEYS})
                current_fields = {key: "" for key in ORDERED_KEYS}
                current_body_lines = []

            block_started = True
            seen_from = True
            in_chain = True

            # Extract inline headers (multiple headers on one line)
            current_fields = extract_inline_headers(stripped, current_fields)

            i += 1
            continue

        # 2. Detect "To:" header line — finalize previous block, start new block
        elif re.match(r"^>*\s*to:", stripped, re.IGNORECASE):
            if block_started:
                current_fields["Body"] = "\n".join(current_body_lines).strip()
                body_chain.append({k: current_fields.get(k, "") for k in ORDERED_KEYS})
                current_fields = {key: "" for key in ORDERED_KEYS}
                current_body_lines = []

            block_started = True
            seen_to = True
            in_chain = True

            # Parse To header value, including multiline with your existing logic
            # (You can add capture_multiline_header_value call here)

            # Example:
            key = "To"
            value = stripped[3:].strip()
            multi_value, next_i = capture_multiline_header_value(lines, i + 1, HEADER_KEYS)
            if multi_value:
                value += " " + multi_value
            current_fields[key] = value.strip()
            i = next_i
            continue

        # 3. Detect other headers (Cc, Bcc, Sent, Subject, Re) — start block if none started
        elif any(re.match(rf"^>*\s*{h}:", stripped, re.IGNORECASE) for h in ["cc", "bcc", "sent", "subject", "re"]):
            if not block_started:
                block_started = True
                in_chain = True

            key = stripped.split(":", 1)[0].capitalize()
            value = stripped.split(":", 1)[1].strip()

            if key.lower() in ["subject", "re"]:
                current_fields[key] = value
                i += 1
                continue
            else:
                multi_value, next_i = capture_multiline_header_value(lines, i + 1, HEADER_KEYS)
                if multi_value:
                    value += " " + multi_value
                current_fields[key] = value.strip()
                i = next_i
                continue

        # 4. Lines part of the current block's body
        elif in_chain:
            current_body_lines.append(line)
            i += 1

        # 5. Lines outside any detected header block, normal body lines
        else:
            regular_body_lines.append(line)
            i += 1

    # After loop, finalize the last block if any
    if block_started:
        current_fields["Body"] = "\n".join(current_body_lines).strip()
        body_chain.append({k: current_fields.get(k, "") for k in ORDERED_KEYS})

    if source_file:
        logging.info(f"Parsed {len(body_chain)} BodyChain blocks from: {source_file}")

    return "\n".join(regular_body_lines).strip(), body_chain


def process_first_n_emails(input_path, output_path, limit=50):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output = []
        for idx, email in enumerate(data[:limit]):
            if "Body" in email:
                source_file = email.get("SourceFile", f"Record_{idx}")
                logging.info(f"[{idx + 1}] Processing {source_file}")
                cleaned_body, body_chain = parse_body_chain_blocks(email["Body"], source_file)
                email["Body"] = cleaned_body
                if body_chain:
                    email["BodyChain"] = body_chain
                else:
                    logging.info(f"[{idx + 1}] No BodyChain detected.")
            output.append(email)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Saved {limit} records to {output_path}")

    except Exception as e:
        logging.error(f"❌ Error: {e}")


if __name__ == "__main__":
    process_first_n_emails(
        input_path=CLEANED_JSON_PATH_1,
        output_path=BODY_CHAIN_OUTPUT_PATH,
        limit=50
    )
