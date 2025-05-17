import json
import re
import logging
from src.config.config import REFINED_JSON_PATH, BODY_CHAIN_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:"]
ORDERED_KEYS = ["From", "To", "Cc", "Bcc", "Subject", "Body"]


def parse_body_chain_blocks(body_text, source_file=None):
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

        is_header = any(lower_line.startswith(h) for h in HEADER_KEYS)

        if lower_line.startswith("from:"):
            if block_started:
                current_fields["Body"] = "\n".join(current_body_lines).strip()
                body_chain.append({k: current_fields.get(k, "") for k in ORDERED_KEYS})
                current_fields = {key: "" for key in ORDERED_KEYS}
                current_body_lines = []
            in_chain = True
            block_started = True
            seen_from = True

            key = "From"
            value = stripped[5:].strip()
            current_fields[key] = value
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()
                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break
                current_fields[key] += " " + next_line
                i += 1
            continue

        elif lower_line.startswith("to:"):
            if block_started and not seen_from:
                current_fields["Body"] = "\n".join(current_body_lines).strip()
                body_chain.append({k: current_fields.get(k, "") for k in ORDERED_KEYS})
                current_fields = {key: "" for key in ORDERED_KEYS}
                current_body_lines = []
            in_chain = True
            block_started = True

            key = "To"
            value = stripped[3:].strip()
            current_fields[key] = value
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()
                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break
                current_fields[key] += " " + next_line
                i += 1
            continue

        elif in_chain and is_header:
            key = lower_line.split(":", 1)[0].capitalize()
            value = stripped.split(":", 1)[1].strip()
            current_fields[key] = value
            i += 1

            if key.lower() in ["subject", "re"]:
                continue

            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()
                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break
                current_fields[key] += " " + next_line
                i += 1
            continue

        elif in_chain:
            current_body_lines.append(line)
            i += 1
        else:
            regular_body_lines.append(line)
            i += 1

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
        input_path=REFINED_JSON_PATH,
        output_path=BODY_CHAIN_OUTPUT_PATH,
        limit=50
    )
