import json
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_cleaning.log'),
        logging.StreamHandler()
    ]
)


def clean_email_body(body):
    """Clean email body by removing unwanted characters and leading spaces."""
    if not body:
        return ""

    original_length = len(body)
    cleaning_steps = []

    # Remove quoted-printable encoding
    body = body.replace('=\n', '')
    body = body.replace('=20', ' ')
    body = body.replace('=09', '\t')
    if len(body) != original_length:
        cleaning_steps.append("Removed quoted-printable encoding")

    # Remove technical headers
    before_headers = len(body)
    body = re.sub(r'Message-ID:.*?\n', '', body)
    body = re.sub(r'X-.*?:\s*.*?\n', '', body, flags=re.MULTILINE)
    body = re.sub(r'Mime-Version:.*?\n', '', body)
    body = re.sub(r'Content-Type:.*?\n', '', body)
    body = re.sub(r'Content-Transfer-Encoding:.*?\n', '', body)
    if len(body) != before_headers:
        cleaning_steps.append("Removed technical headers")

    # Remove forwarded message separators
    before_forwarded = len(body)
    body = re.sub(r'-+\s*Forwarded by.*?\n', '', body)
    body = re.sub(r'---------------------- Forwarded by.*?---------------------------\n', '', body)
    if len(body) != before_forwarded:
        cleaning_steps.append("Removed forwarded message separators")

    # Remove "Original Message" separators
    before_original = len(body)
    body = re.sub(r'-+\s*Original Message\s*-+\n', '', body, flags=re.IGNORECASE)
    body = re.sub(r'-----Original Message-----\n', '', body, flags=re.IGNORECASE)
    if len(body) != before_original:
        cleaning_steps.append("Removed 'Original Message' separators")

    # Remove other common separators
    before_separators = len(body)
    body = re.sub(r'^-{10,}.*?\n', '', body, flags=re.MULTILINE)
    body = re.sub(r'^={10,}.*?\n', '', body, flags=re.MULTILINE)
    if len(body) != before_separators:
        cleaning_steps.append("Removed dash/equals separators")

    # Split into lines and clean each line
    lines = body.split('\n')
    cleaned_lines = []
    leading_spaces_removed = 0

    for line in lines:
        original_line = line
        # Remove leading and trailing whitespace from each line
        cleaned_line = line.strip()

        if len(original_line) > len(cleaned_line):
            leading_spaces_removed += 1

        # Skip completely empty lines at the beginning
        if not cleaned_line and not cleaned_lines:
            continue

        # Add the cleaned line (or empty string for blank lines)
        cleaned_lines.append(cleaned_line)

    if leading_spaces_removed > 0:
        cleaning_steps.append(f"Removed leading/trailing spaces from {leading_spaces_removed} lines")

    # Join lines back together
    result = '\n'.join(cleaned_lines)

    # Remove attachment references
    before_attachments = len(result)
    result = re.sub(r'\s*-\s*.*?\.(doc|pdf|xls|txt|htm|html).*?\n', '', result, flags=re.IGNORECASE)
    result = re.sub(r'<<.*?\.(doc|pdf|xls|txt|htm|html)>>', '', result, flags=re.IGNORECASE)
    if len(result) != before_attachments:
        cleaning_steps.append("Removed attachment references")

    # Clean up excessive blank lines
    before_blanks = len(result)
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    if len(result) != before_blanks:
        cleaning_steps.append("Cleaned up excessive blank lines")

    # Clean up multiple spaces within lines
    before_spaces = len(result)
    result = re.sub(r' {2,}', ' ', result)
    if len(result) != before_spaces:
        cleaning_steps.append("Cleaned up multiple spaces")

    final_length = len(result.strip())
    reduction_percent = round(((original_length - final_length) / original_length) * 100,
                              2) if original_length > 0 else 0

    return result.strip(), cleaning_steps, original_length, final_length, reduction_percent


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
    """Main function to clean Enron emails while preserving all original fields."""

    logging.info(f"Starting email cleaning process")
    logging.info(f"Input file: {json_file_path}")
    logging.info(f"Output file: {output_file_path}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            emails = json.load(file)
        logging.info(f"Successfully loaded {len(emails)} emails from input file")
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return []

    cleaned_emails = []
    total_emails = len(emails)
    processed_count = 0
    skipped_count = 0

    for i, email in enumerate(emails, 1):
        try:
            # Extract email identification info
            message_id = email.get("Message-ID", "Unknown")
            source_file = email.get("SourceFile", "Unknown")
            subject = email.get("Subject", "No Subject")
            from_addr = email.get("From", "Unknown")
            date = email.get("Date", "Unknown")

            logging.info(f"\n{'=' * 60}")
            logging.info(f"Processing Email {i}/{total_emails}")
            logging.info(f"Message ID: {message_id}")
            logging.info(f"Source File: {source_file}")
            logging.info(f"Subject: {subject}")
            logging.info(f"From: {from_addr}")
            logging.info(f"Date: {date}")

            # Create a copy of the original email to preserve all fields
            cleaned_email = email.copy()

            # Clean the email body
            original_body = email.get('Body', '')
            original_body_length = len(original_body)

            logging.info(f"Original body length: {original_body_length} characters")

            if original_body_length == 0:
                logging.warning(f"Email {i} has empty body - SKIPPED")
                skipped_count += 1
                continue

            # Perform cleaning
            cleaned_body, cleaning_steps, orig_len, final_len, reduction_percent = clean_email_body(original_body)

            # Log cleaning steps
            if cleaning_steps:
                logging.info("Cleaning steps performed:")
                for step in cleaning_steps:
                    logging.info(f"  - {step}")
            else:
                logging.info("No cleaning steps needed")

            logging.info(f"Body length after cleaning: {final_len} characters")
            logging.info(f"Size reduction: {reduction_percent}%")

            # Skip if body is too short after cleaning
            if len(cleaned_body.strip()) < 10:
                logging.warning(
                    f"Email {i} body too short after cleaning ({len(cleaned_body.strip())} chars) - SKIPPED")
                skipped_count += 1
                continue

            # Update the Body field with cleaned content
            cleaned_email['Body'] = cleaned_body
            cleaned_emails.append(cleaned_email)
            processed_count += 1

            logging.info(f"Email {i} successfully processed and added to output")

        except Exception as e:
            logging.error(f"Error processing email {i} from {source_file}: {e}")
            skipped_count += 1
            continue

    # Save to JSON with all original fields preserved
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(cleaned_emails, output_file, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved cleaned emails to {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")
        return []

    # Final summary
    logging.info(f"\n{'=' * 60}")
    logging.info("CLEANING PROCESS SUMMARY")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total emails in input: {total_emails}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped emails: {skipped_count}")
    logging.info(f"Success rate: {round((processed_count / total_emails) * 100, 2)}%")
    logging.info(f"Output file: {output_file_path}")
    logging.info("Process completed!")

    return cleaned_emails


# Usage
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron_emails_All_3.json"

    print("Starting Enron email cleaning process...")
    print("Check 'email_cleaning.log' for detailed logs")

    cleaned_data = clean_enron_emails(input_file, output_file)

    print(f"\nProcess completed! Check the log file for details.")
    print(f"Processed {len(cleaned_data)} emails successfully.")