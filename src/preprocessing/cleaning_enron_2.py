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


def validate_email_data(email, email_number):
    """Validate if email has either subject or body content."""
    validation_errors = []

    # Get subject and body
    subject = email.get('Subject', '').strip()
    body = email.get('Body', '').strip()

    # Skip only if BOTH subject and body are empty
    if not subject and not body:
        validation_errors.append("Both subject and body are empty")

    return validation_errors


def clean_email_body(body):
    """Clean email body by removing unwanted characters and symbols."""
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

    # Remove email reply/forward symbols and unwanted artifacts
    before_reply_symbols = len(body)

    # Remove ">" symbols at the beginning of lines (email reply quotes)
    body = re.sub(r'^>\s*', '', body, flags=re.MULTILINE)
    body = re.sub(r'^>\s*>\s*', '', body, flags=re.MULTILINE)  # Multiple levels
    body = re.sub(r'^>\s*>\s*>\s*', '', body, flags=re.MULTILINE)  # Even more levels

    # Remove common email client artifacts
    body = re.sub(r'^\[mailto:.*?\]', '', body, flags=re.MULTILINE)
    body = re.sub(r'<mailto:.*?>', '', body)

    # Remove HTML-like tags if any
    body = re.sub(r'<[^>]+>', '', body)

    # Remove excessive punctuation and symbols
    body = re.sub(r'[*]{3,}', '', body)  # Remove *** patterns
    body = re.sub(r'[_]{3,}', '', body)  # Remove ___ patterns
    body = re.sub(r'[~]{3,}', '', body)  # Remove ~~~ patterns
    body = re.sub(r'[#]{3,}', '', body)  # Remove ### patterns

    if len(body) != before_reply_symbols:
        cleaning_steps.append("Removed email reply symbols and unwanted artifacts")

    # Split into lines and clean each line
    lines = body.split('\n')
    cleaned_lines = []
    leading_spaces_removed = 0

    for line in lines:
        original_line = line
        # Remove leading and trailing whitespace from each line
        cleaned_line = line.strip()

        # Skip lines that are just symbols or empty (but keep meaningful content)
        if re.match(r'^[>\s\-=_*~#]*$', cleaned_line):
            continue

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
    skipped_emails = []  # Track skipped emails
    total_emails = len(emails)
    processed_count = 0
    skipped_count = 0
    validation_skipped = 0
    processing_skipped = 0

    for i, email in enumerate(emails, 1):
        try:
            # Extract email identification info (safely)
            message_id = email.get("Message-ID", "Unknown")
            source_file = email.get("SourceFile", "Unknown")
            subject = email.get("Subject", "").strip()
            from_addr = email.get("From", "Unknown")
            date = email.get("Date", "Unknown")
            body = email.get("Body", "").strip()

            # Calculate stats
            body_length = len(body)
            subject_length = len(subject)
            has_subject = bool(subject)
            has_body = bool(body)

            logging.info(f"\n{'=' * 60}")
            logging.info(f"Processing Email {i}/{total_emails}")
            logging.info(f"Message ID: {message_id}")
            logging.info(f"Source File: {source_file}")
            logging.info(f"Subject: {subject if subject else '[EMPTY]'}")
            logging.info(f"From: {from_addr}")
            logging.info(f"Date: {date}")
            logging.info(f"Has Subject: {has_subject} (length: {subject_length})")
            logging.info(f"Has Body: {has_body} (length: {body_length})")

            # VALIDATION STEP - Check if email has either subject or body
            validation_errors = validate_email_data(email, i)

            if validation_errors:
                skip_reason = f"Validation failed: {', '.join(validation_errors)}"
                logging.warning(f"Email {i} failed validation - SKIPPED")
                logging.warning(f"Validation errors: {', '.join(validation_errors)}")

                skipped_emails.append({
                    "email_number": i,
                    "message_id": message_id,
                    "source_file": source_file,
                    "subject": subject if subject else "[EMPTY]",
                    "from": from_addr,
                    "date": date,
                    "skip_reason": skip_reason,
                    "validation_errors": validation_errors,
                    "has_subject": has_subject,
                    "has_body": has_body,
                    "subject_length": subject_length,
                    "body_length": body_length,
                    "skip_stage": "validation"
                })
                skipped_count += 1
                validation_skipped += 1
                continue

            logging.info("Email passed validation (has subject or body) - proceeding with cleaning")

            # Create a copy of the original email to preserve all fields
            cleaned_email = email.copy()

            # Clean the email body (even if it's empty)
            original_body = email.get('Body', '')
            original_body_length = len(original_body)

            logging.info(f"Original body length: {original_body_length} characters")

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

            # Update the Body field with cleaned content (even if empty)
            cleaned_email['Body'] = cleaned_body
            cleaned_emails.append(cleaned_email)
            processed_count += 1

            logging.info(f"Email {i} successfully processed and added to output")

        except Exception as e:
            skip_reason = f"Processing error: {str(e)}"
            logging.error(f"Error processing email {i} from {source_file}: {e}")
            skipped_emails.append({
                "email_number": i,
                "message_id": email.get("Message-ID", "Unknown"),
                "source_file": email.get("SourceFile", "Unknown"),
                "subject": email.get("Subject", "").strip() or "[EMPTY]",
                "from": email.get("From", "Unknown"),
                "date": email.get("Date", "Unknown"),
                "skip_reason": skip_reason,
                "has_subject": bool(email.get("Subject", "").strip()),
                "has_body": bool(email.get("Body", "").strip()),
                "subject_length": len(email.get("Subject", "").strip()),
                "body_length": len(email.get("Body", "").strip()),
                "skip_stage": "processing_error"
            })
            skipped_count += 1
            processing_skipped += 1
            continue

    # Save to JSON with all original fields preserved
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(cleaned_emails, output_file, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved cleaned emails to {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")
        return []

    # Save skipped emails report
    if skipped_emails:
        skipped_report_file = output_file_path.replace('.json', '_skipped_report.json')
        try:
            with open(skipped_report_file, 'w', encoding='utf-8') as skipped_file:
                json.dump(skipped_emails, skipped_file, indent=2, ensure_ascii=False)
            logging.info(f"Skipped emails report saved to {skipped_report_file}")
        except Exception as e:
            logging.error(f"Failed to save skipped emails report: {e}")

    # Final summary
    logging.info(f"\n{'=' * 60}")
    logging.info("CLEANING PROCESS SUMMARY")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total emails in input: {total_emails}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped emails: {skipped_count}")
    logging.info(f"  - Skipped at validation (no subject AND no body): {validation_skipped}")
    logging.info(f"  - Skipped due to processing errors: {processing_skipped}")
    logging.info(f"Success rate: {round((processed_count / total_emails) * 100, 2)}%")

    # Detailed skipped emails summary
    if skipped_emails:
        logging.info(f"\n{'=' * 60}")
        logging.info("SKIPPED EMAILS DETAILS")
        logging.info(f"{'=' * 60}")

        # Group by skip stage and reason
        skip_stages = {}
        for skipped in skipped_emails:
            stage = skipped.get('skip_stage', 'unknown')
            if stage not in skip_stages:
                skip_stages[stage] = {}

            reason = skipped['skip_reason']
            if reason not in skip_stages[stage]:
                skip_stages[stage][reason] = []
            skip_stages[stage][reason].append(skipped)

        for stage, reasons in skip_stages.items():
            logging.info(f"\nSKIP STAGE: {stage.upper()}")
            logging.info("=" * 40)

            for reason, emails in reasons.items():
                logging.info(f"\nReason: {reason} ({len(emails)} emails)")
                logging.info("-" * 50)
                for email in emails:
                    logging.info(f"  Email #{email['email_number']}")
                    logging.info(f"    Source File: {email['source_file']}")
                    logging.info(f"    Message ID: {email['message_id']}")
                    logging.info(f"    Subject: {email['subject']}")
                    logging.info(f"    Has Subject: {email.get('has_subject', 'N/A')}")
                    logging.info(f"    Has Body: {email.get('has_body', 'N/A')}")
                    logging.info(f"    Subject Length: {email.get('subject_length', 'N/A')} chars")
                    logging.info(f"    Body Length: {email.get('body_length', 'N/A')} chars")
                    if 'validation_errors' in email:
                        logging.info(f"    Validation Errors: {', '.join(email['validation_errors'])}")
                    logging.info("")

    logging.info(f"Output file: {output_file_path}")
    if skipped_emails:
        logging.info(f"Skipped emails report: {output_file_path.replace('.json', '_skipped_report.json')}")
    logging.info("Process completed!")

    return cleaned_emails


# Usage
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron_emails_All_7.json"

    print("Starting Enron email cleaning process...")
    print("Check 'email_cleaning.log' for detailed logs")

    cleaned_data = clean_enron_emails(input_file, output_file)

    print(f"\nProcess completed! Check the log file for details.")
    print(f"Processed {len(cleaned_data)} emails successfully.")
    print("Check 'cleaned_enron_emails_skipped_report.json' for details on skipped emails.")