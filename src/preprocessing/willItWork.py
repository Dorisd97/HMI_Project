import json
import re
import logging
from src.config.config import CLEANED_JSON_PATH, BODY_CHAIN_OUTPUT_PATH

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:"]

def parse_body_chain_blocks(body_text, source_file=None):
    lines = body_text.splitlines()
    body_chain = []
    regular_body_lines = []

    current_block = {}
    block_body_lines = []
    current_header = None
    in_block = False
    encountered_first_header = False

    def flush_block():
        if current_block or block_body_lines:
            block = current_block.copy()
            block["Body"] = "\n".join(block_body_lines).strip()
            body_chain.append(block)
            current_block.clear()
            block_body_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lower_line = stripped.lower()

        is_header = any(lower_line.startswith(h) for h in HEADER_KEYS)

        if lower_line.startswith("from:") or lower_line.startswith("to:"):
            if in_block:
                flush_block()
            in_block = True
            encountered_first_header = True

            current_header = lower_line.split(":", 1)[0].capitalize()
            current_block[current_header] = stripped.split(":", 1)[1].strip()

            # Handle header wrapping for From/To
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()

                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break

                # allow wrapping only for non-subject headers
                if current_header.lower() not in ["subject", "re"] and next_line:
                    current_block[current_header] += " " + next_line.strip()
                else:
                    break
                i += 1
            continue

        elif in_block and is_header:
            current_header = lower_line.split(":", 1)[0].capitalize()
            current_block[current_header] = stripped.split(":", 1)[1].strip()

            # Subject: take only first line
            if current_header.lower() in ["subject", "re"]:
                i += 1
                continue

            # Other headers may wrap
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()
                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break
                if next_line:
                    current_block[current_header] += " " + next_line.strip()
                else:
                    break
                i += 1
            continue

        elif in_block:
            block_body_lines.append(line)
            i += 1

        else:
            regular_body_lines.append(line)
            i += 1

    if in_block:
        flush_block()

    if source_file:
        logging.info(f"Parsed {len(body_chain)} chains from: {source_file}")

    return "\n".join(regular_body_lines).strip(), body_chain


def process_first_n_emails(input_path, output_path, limit=50):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output = []
        for idx, email in enumerate(data[:limit]):
            if "Body" in email:
                source_file = email.get("SourceFile", f"Record_{idx}")
                logging.info(f"[{idx+1}] Processing {source_file}")
                cleaned_body, body_chain = parse_body_chain_blocks(email["Body"], source_file)
                email["Body"] = cleaned_body
                if body_chain:
                    email["BodyChain"] = body_chain
                else:
                    logging.info(f"[{idx+1}] No BodyChain detected.")
            output.append(email)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Successfully saved {limit} processed records to: {output_path}")

    except Exception as e:
        logging.error(f"❌ Error processing file: {e}")

# Main entrypoint
if __name__ == "__main__":
    process_first_n_emails(
        input_path=CLEANED_JSON_PATH,
        output_path=BODY_CHAIN_OUTPUT_PATH,
        limit=50
    )
