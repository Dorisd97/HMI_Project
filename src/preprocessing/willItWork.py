import json
import re

HEADER_KEYS = ["from:", "to:", "cc:", "bcc:", "subject:", "re:"]

def parse_body_chain_blocks(body_text):
    lines = body_text.splitlines()
    body_chain = []
    regular_body_lines = []

    current_block = {}
    block_body_lines = []
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
        line = lines[i].strip()
        lower_line = line.lower()

        if lower_line.startswith("from:") or lower_line.startswith("to:"):
            if in_block:
                flush_block()
            else:
                encountered_first_header = True
                in_block = True

            current_block = {}
            if lower_line.startswith("from:"):
                current_block["From"] = line[5:].strip()
            elif lower_line.startswith("to:"):
                current_block["To"] = line[3:].strip()
            i += 1

            # Handle other headers
            while i < len(lines):
                peek = lines[i].strip()
                peek_lower = peek.lower()

                if peek_lower.startswith("from:") or peek_lower.startswith("to:"):
                    break
                elif peek_lower.startswith("cc:"):
                    current_block["Cc"] = peek[3:].strip()
                elif peek_lower.startswith("bcc:"):
                    current_block["Bcc"] = peek[4:].strip()
                elif peek_lower.startswith("subject:") or peek_lower.startswith("re:"):
                    current_block["Subject"] = re.sub(r'^(subject:|re:)', '', peek, flags=re.I).strip()
                    i += 1
                    break  # body starts after subject
                i += 1

            # Start collecting body lines after headers
            while i < len(lines):
                content_line = lines[i].strip()
                if content_line.lower().startswith("from:") or content_line.lower().startswith("to:"):
                    break
                block_body_lines.append(lines[i])
                i += 1

        else:
            if not encountered_first_header:
                regular_body_lines.append(lines[i])
            else:
                # For lines in between misformatted or extra lines
                if in_block:
                    block_body_lines.append(lines[i])
            i += 1

    if in_block:
        flush_block()

    cleaned_body = "\n".join(regular_body_lines).strip()
    return cleaned_body, body_chain

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

        print(f"✅ Fixed and saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Run it
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data_with_body_chain.json"
    process_email_json(input_file, output_file)
