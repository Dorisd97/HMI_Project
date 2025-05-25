import json
import re
from datetime import datetime


def parse_email_thread(body):
    """Parse email body to automatically detect and format email threads."""
    if not body:
        return ""

    # Remove quoted-printable encoding first
    body = body.replace('=\n', '')
    body = body.replace('=20', ' ')
    body = body.replace('=09', '\t')

    # Split into lines for processing
    lines = body.split('\n')
    formatted_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip technical headers
        if re.match(r'^(Message-ID:|X-|Mime-Version:|Content-Type:|Content-Transfer-Encoding:)', line):
            i += 1
            continue

        # Skip forwarded message separators
        if re.match(r'^-+\s*Forwarded by.*?-+$', line):
            i += 1
            continue

        # Detect email metadata patterns
        if re.match(r'^.+@.+\s+on\s+\d+/\d+/\d+', line):
            # This is sender info - add separator and format
            formatted_lines.append('----------------------------')
            formatted_lines.append(line)

            # Look for To:, cc:, Subject: lines that follow
            j = i + 1
            while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                next_line = lines[j].strip()
                if re.match(r'^(To:|cc:|Subject:)', next_line, re.IGNORECASE):
                    formatted_lines.append(next_line)
                    j += 1
                elif next_line == "":
                    j += 1
                else:
                    break

            formatted_lines.append('----------------------------')
            i = j
            continue

        # Detect email signature patterns (name followed by contact info)
        elif re.match(r'^[A-Za-z\s]+$', line) and len(line.split()) <= 3:
            # Check if next few lines contain contact info
            potential_signature = [line]
            j = i + 1
            contact_found = False

            while j < len(lines) and j < i + 5:
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                # Look for phone numbers, email addresses, or job titles
                if (re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', next_line) or
                        re.search(r'[\w\.-]+@[\w\.-]+\.\w+', next_line) or
                        any(title in next_line.lower() for title in
                            ['vice president', 'director', 'manager', 'analyst'])):
                    contact_found = True
                    potential_signature.append(next_line)
                    j += 1
                elif len(next_line.split()) <= 4:  # Short line, might be continuation
                    potential_signature.append(next_line)
                    j += 1
                else:
                    break

            if contact_found:
                formatted_lines.extend(potential_signature)
                i = j
                continue

        # Regular content line
        if line:
            formatted_lines.append(line)
        elif formatted_lines and formatted_lines[-1] != "":
            formatted_lines.append("")  # Preserve paragraph breaks

        i += 1

    # Join and clean up
    result = '\n'.join(formatted_lines)

    # Clean up excessive blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def clean_email_body(body):
    """Clean email body while preserving thread structure."""
    if not body:
        return ""

    # First, try to parse as email thread
    parsed_thread = parse_email_thread(body)

    if parsed_thread:
        # Additional cleaning
        cleaned = parsed_thread

        # Remove attachment references
        cleaned = re.sub(r'\s*-\s*.*?\.(doc|pdf|xls|txt|htm|html).*?\n', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<<.*?\.(doc|pdf|xls|txt|htm|html)>>', '', cleaned, flags=re.IGNORECASE)

        # Clean up multiple spaces
        cleaned = re.sub(r' {3,}', ' ', cleaned)

        # Fix common formatting issues
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)

        return cleaned.strip()

    return body.strip()


def detect_thread_separators(body):
    """Detect natural separators in email threads."""
    separators = []
    lines = body.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        # Patterns that typically indicate start of new email in thread
        patterns = [
            r'^.+@.+\s+\d{2}/.+',  # email@domain MM/DD/YYYY pattern
            r'^\w+\s+\w+@\w+$',  # Name@Company pattern
            r'^From:\s*.+@',  # From: email pattern
            r'^-+\s*Original Message\s*-+',  # Original message separator
        ]

        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                separators.append(i)
                break

    return separators


def smart_email_formatting(body):
    """Apply smart formatting based on detected patterns."""
    if not body:
        return ""

    lines = body.split('\n')
    formatted_lines = []

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Add separators before email metadata
        if (re.match(r'^.+@.+\s+on\s+\d+/\d+/\d+', stripped_line) or
                re.match(r'^\w+\s+\w+@\w+\s+\d+/\d+/\d+', stripped_line)):

            if formatted_lines and formatted_lines[-1] != '----------------------------':
                formatted_lines.append('----------------------------')
            formatted_lines.append(stripped_line)

            # Check next lines for To:, cc:, Subject:
            j = i + 1
            while j < len(lines) and j < i + 5:
                next_line = lines[j].strip()
                if re.match(r'^(To:|cc:|Subject:)', next_line, re.IGNORECASE):
                    formatted_lines.append(next_line)
                elif next_line == '':
                    pass  # Skip empty lines in metadata section
                else:
                    break
                j += 1

            formatted_lines.append('----------------------------')
            continue

        # Regular line processing
        if stripped_line:
            formatted_lines.append(stripped_line)
        elif formatted_lines and formatted_lines[-1] != '':
            formatted_lines.append('')  # Preserve paragraph breaks

    # Join and final cleanup
    result = '\n'.join(formatted_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def extract_name_from_email(email_addr):
    """Extract name from email address."""
    if not email_addr:
        return ""

    email_addr = re.sub(r'[<>]', '', email_addr)

    # Extract name if present
    name_match = re.search(r'^(.*?)<', email_addr)
    if name_match:
        return name_match.group(1).strip().strip('"')

    # Extract from email address
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
            # Apply smart formatting to preserve email threads
            original_body = email.get('Body', '')
            cleaned_body = smart_email_formatting(original_body)

            # Further cleaning
            cleaned_body = clean_email_body(cleaned_body)

            # Skip if body is too short
            if len(cleaned_body.strip()) < 10:
                print(f"Skipping email {i + 1}: Body too short after cleaning")
                continue

            # Create cleaned email object
            cleaned_email = {
                "ID": extract_message_id(email.get("Message-ID", "")),
                "Date": format_date(email.get("Date", "")),
                "From": extract_name_from_email(email.get("From", "")),
                "From_email": clean_email_address(email.get("From", "")),
                "To": clean_email_address(email.get("To", "")),
                "Subject": email.get("Subject", "").strip(),
                "Body": cleaned_body
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
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron_emails_CA.json"

    clean_enron_emails(input_file, output_file)