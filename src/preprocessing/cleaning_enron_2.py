import json
import re
from datetime import datetime


def clean_email_body(body):
    """Clean email body by removing unwanted characters and leading spaces."""
    if not body:
        return ""

    # Remove quoted-printable encoding
    body = body.replace('=\n', '')
    body = body.replace('=20', ' ')
    body = body.replace('=09', '\t')

    # Remove technical headers
    body = re.sub(r'Message-ID:.*?\n', '', body)
    body = re.sub(r'X-.*?:\s*.*?\n', '', body, flags=re.MULTILINE)
    body = re.sub(r'Mime-Version:.*?\n', '', body)
    body = re.sub(r'Content-Type:.*?\n', '', body)
    body = re.sub(r'Content-Transfer-Encoding:.*?\n', '', body)

    # Remove forwarded message separators
    body = re.sub(r'-+\s*Forwarded by.*?\n', '', body)
    body = re.sub(r'---------------------- Forwarded by.*?---------------------------\n', '', body)
    body = re.sub(r'^-{10,}.*?\n', '', body, flags=re.MULTILINE)
    body = re.sub(r'^={10,}.*?\n', '', body, flags=re.MULTILINE)

    # Split into lines and clean each line
    lines = body.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove leading and trailing whitespace from each line
        cleaned_line = line.strip()

        # Skip completely empty lines at the beginning
        if not cleaned_line and not cleaned_lines:
            continue

        # Add the cleaned line (or empty string for blank lines)
        cleaned_lines.append(cleaned_line)

    # Join lines back together
    result = '\n'.join(cleaned_lines)

    # Remove attachment references
    result = re.sub(r'\s*-\s*.*?\.(doc|pdf|xls|txt|htm|html).*?\n', '', result, flags=re.IGNORECASE)
    result = re.sub(r'<<.*?\.(doc|pdf|xls|txt|htm|html)>>', '', result, flags=re.IGNORECASE)

    # Clean up excessive blank lines (more than 2 consecutive)
    result = re.sub(r'\n{4,}', '\n\n\n', result)

    # Clean up multiple spaces within lines
    result = re.sub(r' {2,}', ' ', result)

    return result.strip()


def extract_name_from_email(email_addr):
    """Extract name from email address."""
    if not email_addr:
        return ""

    email_addr = re.sub(r'[<>]', '', email_addr)

    name_match = re.search(r'^(.*?)<', email_addr)
    if name_match:
        return name_match.group(1).strip().strip('"')

    local_part = email_addr.split('@')[0]

    if '.' in local_part:
        parts = local_part.split('.')
        return ' '.join(part.title() for part in parts[:2])

    return local_part.title()


def format_date(date_str):
    """Format date string consistently."""
    try:
        if '.' in date_str and len(date_str.split('.')) == 3:
            dt = datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
            return dt.strftime("%m/%d/%Y %I:%M:%S %p")
        else:
            return date_str
    except:
        return date_str


def clean_email_address(email_str):
    """Clean and extract email address."""
    if not email_str:
        return ""

    email_str = re.sub(r'\s+', ' ', email_str.strip())
    email_str = re.sub(r'[<>"\']', '', email_str)

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_str)
    if email_match:
        return email_match.group(0)

    return email_str.strip()


def extract_message_id(message_id_str):
    """Extract clean message ID."""
    if not message_id_str:
        return ""

    clean_id = re.sub(r'[<>]', '', message_id_str)
    return clean_id.strip()


def clean_enron_emails(json_file_path, output_file_path):
    """Main function to clean and format Enron emails."""

    with open(json_file_path, 'r', encoding='utf-8') as file:
        emails = json.load(file)

    cleaned_emails = []

    for i, email in enumerate(emails):
        try:
            # Clean the email body
            original_body = email.get('Body', '')
            cleaned_body = clean_email_body(original_body)

            # Skip if body is too short
            if len(cleaned_body.strip()) < 10:
                print(f"Skipping email {i + 1}: Body too short after cleaning")
                continue

            # Create cleaned email object
            cleaned_email = {
                "id": extract_message_id(email.get("Message-ID", "")),
                "date": format_date(email.get("Date", "")),
                "from": extract_name_from_email(email.get("From", "")),
                "from_email": clean_email_address(email.get("From", "")),
                "to": clean_email_address(email.get("To", "")),
                "subject": email.get("Subject", "").strip(),
                "body": cleaned_body
            }

            cleaned_emails.append(cleaned_email)

        except Exception as e:
            print(f"Error processing email {i + 1}: {e}")
            continue

    # Save to JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(cleaned_emails, output_file, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(cleaned_emails)} out of {len(emails)} emails")
    print(f"Cleaned emails saved to {output_file_path}")

    return cleaned_emails


# Usage
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_50_Data.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron_emails_2.json"

    clean_enron_emails(input_file, output_file)