import json
import re
import logging
from typing import List, Dict, Tuple, Any
from src.config.config import CLEANED_JSON_PATH, BODY_CHAIN_OUTPUT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Enhanced header detection patterns
HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:", "sent:", "date:", "forwarded:", "original message:"]
ORDERED_KEYS = ["From", "Sent", "To", "Cc", "Bcc", "Subject", "Body"]
CHAIN_HEADER_KEYS = ['To', 'Cc', 'Bcc', 'Subject', 'Sent', 'From']
CONTINUATION_RE = re.compile(r'^[ \t]+')  # lines starting with space or tab

# Patterns for detecting email thread boundaries
THREAD_BOUNDARY_PATTERNS = [
    r"^-+\s*original\s+message\s*-+",
    r"^-+\s*forwarded\s+message\s*-+",
    r"^>+\s*from:",
    r"^from:",
    r"^to:",
    r"^\s*>{3,}",  # quoted text with multiple > symbols
]


def is_thread_boundary(line: str) -> bool:
    """Check if a line indicates the start of a new email thread."""
    line_lower = line.lower().strip()
    return any(re.match(pattern, line_lower, re.IGNORECASE) for pattern in THREAD_BOUNDARY_PATTERNS)


def capture_multiline_header_value(lines: List[str], start_idx: int, header_keys: List[str]) -> Tuple[str, int]:
    """
    Capture multiline header values, including proper continuation line handling.
    Returns the captured value and the next line index to process.
    """
    captured_lines = []
    i = start_idx

    # Create regex to detect any header at start of line
    header_re = re.compile(rf"^>*\s*({'|'.join(re.escape(h) for h in header_keys)})",
                           flags=re.IGNORECASE)

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Stop on blank line (start of body)
        if not stripped.strip():
            break

        # Stop on new header
        if header_re.match(stripped):
            break

        # Stop on thread boundary
        if is_thread_boundary(stripped):
            break

        # Must be indented to be a continuation
        if not CONTINUATION_RE.match(line):
            break

        # Valid continuation line
        captured_lines.append(stripped.strip())
        i += 1

    return " ".join(captured_lines), i


def extract_inline_headers(line: str, current_fields: Dict[str, str]) -> Dict[str, str]:
    """Extract multiple headers that might appear on a single line."""
    header_patterns = {
        "From": r'from:\s*(.*?)(?=\s+(?:to|cc|bcc|sent|subject):|$)',
        "Sent": r'sent:\s*(.*?)(?=\s+(?:to|cc|bcc|from|subject):|$)',
        "To": r'to:\s*(.*?)(?=\s+(?:from|cc|bcc|sent|subject):|$)',
        "Cc": r'cc:\s*(.*?)(?=\s+(?:from|to|bcc|sent|subject):|$)',
        "Bcc": r'bcc:\s*(.*?)(?=\s+(?:from|to|cc|sent|subject):|$)',
        "Subject": r'subject:\s*(.*)',  # Subject typically goes to end of line
    }

    for key, pattern in header_patterns.items():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:  # Only update if we found a non-empty value
                current_fields[key] = value

    return current_fields


def normalize_header_key(header_text: str) -> str:
    """Normalize header keys to standard format."""
    clean_key = re.sub(r'^>*\s*', '', header_text).split(':')[0].strip()

    # Map variations to standard keys
    key_mapping = {
        'from': 'From',
        'to': 'To',
        'cc': 'Cc',
        'bcc': 'Bcc',
        'sent': 'Sent',
        'date': 'Sent',
        'subject': 'Subject',
        're': 'Subject',  # Re: is often used as subject prefix
    }

    return key_mapping.get(clean_key.lower(), clean_key.capitalize())


import re

def parse_body_chain_blocks(body_text, source_file=None):
    HEADERS = ["From", "Sent", "To", "Cc", "Bcc", "Subject"]
    header_re = re.compile(r'^(from|sent|to|cc|bcc|subject):', re.IGNORECASE)

    lines = body_text.splitlines()
    n = len(lines)
    i = 0
    main_body_lines = []
    body_chain = []

    def parse_headers(i):
        headers = {k: "" for k in HEADERS}
        while i < n:
            line = lines[i]
            m = header_re.match(line.strip())
            if not m:
                break
            header = m.group(1).capitalize()
            value = line.split(':', 1)[1].strip()
            i += 1
            if header == "Subject":
                # Only use the first line after Subject:
                headers[header] = value
            else:
                # For others, keep grabbing lines until next header or blank
                multi_value = [value]
                while i < n:
                    next_line = lines[i]
                    if header_re.match(next_line.strip()) or not next_line.strip():
                        break
                    multi_value.append(next_line.strip())
                    i += 1
                headers[header] = ' '.join(multi_value).strip()
            # Skip blank lines between headers and body
            while i < n and not lines[i].strip():
                i += 1
        return headers, i

    while i < n:
        if header_re.match(lines[i].strip()):
            headers, i = parse_headers(i)
            body_lines = []
            while i < n and not header_re.match(lines[i].strip()):
                body_lines.append(lines[i])
                i += 1
            block = {k: headers.get(k, "") for k in HEADERS}
            block["Body"] = '\n'.join(body_lines).strip()
            body_chain.append(block)
        else:
            main_body_lines.append(lines[i])
            i += 1

    main_body = '\n'.join(main_body_lines).strip()
    if source_file:
        import logging
        logging.info(f"Parsed {len(body_chain)} BodyChain blocks from: {source_file}")
    return main_body, body_chain


def validate_email_fields(email_block: Dict[str, str]) -> bool:
    """Validate that an email block has minimum required fields."""
    required_fields = ["From", "Subject"]
    return any(email_block.get(field, "").strip() for field in required_fields)


def process_first_n_emails(input_path: str, output_path: str) -> None:
    """
    Process the first N emails from input JSON file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        limit: Number of emails to process
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON array of email objects")

        output = []
        processed_count = 0
        chain_count = 0

        for idx, email in enumerate(data):
            if not isinstance(email, dict):
                logging.warning(f"[{idx + 1}] Skipping non-dict entry")
                continue

            if "Body" not in email:
                logging.warning(f"[{idx + 1}] Skipping email without Body field")
                output.append(email)
                continue

            source_file = email.get("SourceFile", f"Record_{idx}")
            logging.info(f"[{idx + 1}] Processing {source_file}")

            try:
                cleaned_body, body_chain = parse_body_chain_blocks(email["Body"], source_file)

                # Update email with cleaned body
                email["Body"] = cleaned_body

                # Add body chain if any valid chains were found
                valid_chains = [block for block in body_chain if validate_email_fields(block)]
                if valid_chains:
                    email["BodyChain"] = valid_chains
                    chain_count += len(valid_chains)
                    logging.info(f"[{idx + 1}] Found {len(valid_chains)} valid email chain(s)")
                else:
                    logging.info(f"[{idx + 1}] No valid email chains detected")

                processed_count += 1

            except Exception as e:
                logging.error(f"[{idx + 1}] Error processing {source_file}: {e}")

            output.append(email)

        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Successfully processed {processed_count} emails")
        logging.info(f"✅ Total email chains extracted: {chain_count}")
        logging.info(f"✅ Saved results to {output_path}")

    except FileNotFoundError:
        logging.error(f"❌ Input file not found: {input_path}")
    except json.JSONDecodeError as e:
        logging.error(f"❌ Invalid JSON in input file: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    process_first_n_emails(
        input_path=CLEANED_JSON_PATH,
        output_path=BODY_CHAIN_OUTPUT_PATH
    )