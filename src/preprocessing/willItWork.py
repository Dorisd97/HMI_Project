import json
import re


def extract_body_chain_lines(body_text):
    body_chain = []
    current_email = {}
    collecting_body = False
    body_lines = []

    lines = body_text.splitlines()

    for line in lines:
        line_stripped = line.strip()

        # Start of a new email block
        if re.match(r'^From:\s+', line_stripped):
            # Save previous email block if it exists
            if current_email or body_lines:
                current_email["Body"] = "\n".join(body_lines).strip()
                body_chain.append(current_email)
                current_email = {}
                body_lines = []

            current_email["From"] = line_stripped.split(":", 1)[1].strip()
            collecting_body = False

        elif re.match(r'^To:\s+', line_stripped):
            current_email["To"] = line_stripped.split(":", 1)[1].strip()
        elif re.match(r'^cc:\s+', line_stripped, re.IGNORECASE):
            current_email["cc"] = line_stripped.split(":", 1)[1].strip()
        elif re.match(r'^(Subject|Re):\s+', line_stripped):
            current_email["Subject"] = line_stripped.split(":", 1)[1].strip()
        elif re.match(r'^[A-Za-z\-]+:', line_stripped):  # Skip other headers
            continue
        else:
            collecting_body = True
            body_lines.append(line)

    # Append the last message if exists
    if current_email or body_lines:
        current_email["Body"] = "\n".join(body_lines).strip()
        body_chain.append(current_email)

    return body_chain


def process_emails(email_data):
    processed_emails = []

    for email in email_data:
        full_body = email.get("Body", "")
        body_chain = extract_body_chain_lines(full_body)

        # Main body is everything before first "From:" match
        first_from_index = next(
            (i for i, line in enumerate(full_body.splitlines()) if line.strip().startswith("From:")), None)
        main_body = "\n".join(
            full_body.splitlines()[:first_from_index]).strip() if first_from_index is not None else full_body.strip()

        email_object = {
            "Message-ID": email.get("Message-ID"),
            "Date": email.get("Date"),
            "From": email.get("From"),
            "To": email.get("To"),
            "Subject": email.get("Subject"),
            "Mime-Version": email.get("Mime-Version"),
            "Content-Type": email.get("Content-Type"),
            "Content-Transfer-Encoding": email.get("Content-Transfer-Encoding"),
            "X-From": email.get("X-From"),
            "X-To": email.get("X-To"),
            "X-cc": email.get("X-cc"),
            "X-bcc": email.get("X-bcc"),
            "X-Folder": email.get("X-Folder"),
            "X-Origin": email.get("X-Origin"),
            "X-FileName": email.get("X-FileName"),
            "SourceFile": email.get("SourceFile"),
            "Body": main_body,
            "BodyChain": body_chain
        }

        processed_emails.append(email_object)

    return processed_emails


# Load JSON
with open("D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Process
structured_output = process_emails(raw_data)

# Save to output
with open("D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data_processed_emails.json", "w", encoding="utf-8") as f:
    json.dump(structured_output, f, indent=2)
