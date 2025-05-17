import json

def extract_all_to_blocks(body_text):
    lines = body_text.splitlines()
    to_blocks = []
    current_block = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # If a new To: line starts, flush old block if any and start a new one
        if lower.startswith("to:"):
            if current_block:
                to_blocks.append("\n".join(current_block))
                current_block = []
            capturing = True
            current_block.append(stripped)
        elif capturing and (lower.startswith("cc:") or lower.startswith("subject:") or lower.startswith("re:") or lower.startswith("from:") or lower.startswith("to:")):
            # End current block, do NOT include this line
            if current_block:
                to_blocks.append("\n".join(current_block))
                current_block = []
            capturing = False
        elif capturing:
            current_block.append(stripped)

    # Append the last block if it was not closed
    if current_block:
        to_blocks.append("\n".join(current_block))

    return to_blocks

def extract_to_addresses(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, email in enumerate(data):
            print(f"\n===== Email {idx + 1} =====")
            if "Body" in email:
                blocks = extract_all_to_blocks(email["Body"])
                if blocks:
                    for i, block in enumerate(blocks, 1):
                        print(f"\n--- To Block #{i} ---\n{block}")
                else:
                    print("No 'To:' blocks found in body.")
            else:
                print("No 'Body' field found.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json"
    extract_to_addresses(file_path)
