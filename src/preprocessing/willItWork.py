import json
import re

HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:"]

def parse_body_chain_blocks(body_text):
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

        # Case 1: New From: or To: starts a new block
        if lower_line.startswith("from:") or lower_line.startswith("to:"):
            if in_block:
                flush_block()
            in_block = True
            encountered_first_header = True

            current_header = lower_line.split(":", 1)[0].capitalize()
            current_block[current_header] = stripped.split(":", 1)[1].strip()

            # Handle wrapped headers
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                next_lower = next_line.lower()

                # Stop on next known header
                if any(next_lower.startswith(h) for h in HEADER_KEYS):
                    break

                # Continuation (only for From, To, Cc, Bcc)
                if current_header.lower() not in ["subject", "re"] and next_line:
                    current_block[current_header] += " " + next_line.strip()
                else:
                    break
                i += 1
            continue

        # Case 2: Inside a block and encountering additional headers
        elif in_block and is_header:
            current_header = lower_line.split(":", 1)[0].capitalize()
            current_block[current_header] = stripped.split(":", 1)[1].strip()

            # Subject must NOT take continuation lines
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

        # Case 3: Collecting body content
        elif in_block:
            block_body_lines.append(line)
            i += 1

        # Case 4: Top-level pre-chain body
        else:
            regular_body_lines.append(line)
            i += 1

    if in_block:
        flush_block()

    return "\n".join(regular_body_lines).strip(), body_chain

def process_email_json(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for email in data:
            if "Body" in email:
                cleaned_body, body_chain = parse_body_chain_blocks(email["Body"])
                email["Body"] = cleaned_body
                if body_chain:
                    email["BodyChain"] = body_chain

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Final JSON saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Run this
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data_with_body_chain.json"
    process_email_json(input_file, output_file)
