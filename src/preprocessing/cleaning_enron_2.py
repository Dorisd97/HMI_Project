import json
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../log/email_cleaning.log'),
        logging.StreamHandler()
    ]
)


def is_valid_bullet_point(line):
    """Check if a line contains a valid structured list item (numbers/letters only)."""
    if not line.strip():
        return False

    # Only preserve structured, formal list formats - no symbol bullets
    # Check the stripped version to identify the pattern
    stripped_line = line.strip()
    valid_patterns = [
        r'^\d+\.\s+\w+',  # 1. 2. 3. numbered lists
        r'^[a-zA-Z]\.\s+\w+',  # a. b. c. lettered lists
        r'^[a-zA-Z]\)\s+\w+',  # a) b) c. lettered lists with parentheses
        r'^\(\d+\)\s+\w+',  # (1) (2) (3) numbered with parentheses
        r'^\d+\)\s+\w+',  # 1) 2) 3. numbered with closing parenthesis
        r'^[IVX]+\.\s+\w+',  # I. II. III. Roman numerals
    ]

    return any(re.match(pattern, stripped_line, re.IGNORECASE) for pattern in valid_patterns)


def clean_leading_characters(line):
    """Remove ALL leading spaces and symbols from lines."""
    if not line.strip():
        return ""

    original_line = line
    # Always remove ALL leading whitespace first
    cleaned_line = line.lstrip()

    # If it's a valid structured bullet point, keep it as is (but without leading spaces)
    if is_valid_bullet_point(cleaned_line):
        return cleaned_line

    # For all other lines, aggressively remove leading symbols
    if cleaned_line:
        # Remove all common email and symbol artifacts at the beginning
        symbol_patterns_to_remove = [
            r'^[>]+[\s]*',  # Remove > >> >>> email replies
            r'^[-*•·▪▫◦‣⁃]+[\s]*',  # Remove symbol bullets
            r'^[=]+[\s]*',  # Remove = symbols
            r'^[-]+[\s]*',  # Remove - symbols (when not bullet)
            r'^[_]+[\s]*',  # Remove _ symbols
            r'^[*]+[\s]*',  # Remove * symbols
            r'^[~]+[\s]*',  # Remove ~ symbols
            r'^[#]+[\s]*',  # Remove # symbols
            r'^[\.]{3,}[\s]*',  # Remove excessive dots
            r'^[\|]+[\s]*',  # Remove | symbols
            r'^[+]+[\s]*',  # Remove + symbols
        ]

        # Apply symbol removal patterns
        for pattern in symbol_patterns_to_remove:
            if re.match(pattern, cleaned_line):
                cleaned_line = re.sub(pattern, '', cleaned_line)
                break

        # Remove any remaining leading symbols that aren't alphanumeric
        cleaned_line = re.sub(r'^[^\w\s]+[\s]*', '', cleaned_line)

        # Final cleanup - remove any remaining isolated symbols at the start
        if cleaned_line and not re.match(r'^\w', cleaned_line):
            # If line doesn't start with word character, clean more aggressively
            cleaned_line = re.sub(r'^[^\w]+', '', cleaned_line)

    return cleaned_line


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
    """Clean email body by removing symbols, unwanted characters, and ALL leading spaces."""
    if not body:
        return ""

    original_length = len(body)
    cleaning_steps = []
    result = body

    # 1. Clean quoted-printable encoding (these are definitely artifacts)
    before_qp = len(result)
    result = result.replace('=\n', '')  # Remove soft line breaks
    result = result.replace('=20', ' ')  # Replace encoded spaces
    result = result.replace('=09', '\t')  # Replace encoded tabs
    result = result.replace('=3D', '=')  # Replace encoded equals
    result = result.replace('=22', '"')  # Replace encoded quotes
    result = result.replace('=27', "'")  # Replace encoded apostrophes
    if len(result) != before_qp:
        cleaning_steps.append("Cleaned quoted-printable encoding")

    # 2. Clean up excessive whitespace (but preserve structure)
    before_whitespace = len(result)

    # Replace multiple spaces with single space (but preserve intentional spacing)
    result = re.sub(r'[ \t]{3,}', ' ', result)  # Only if 3+ spaces/tabs

    # Clean up excessive blank lines (but keep paragraph breaks)
    result = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', result)  # Max 3 consecutive newlines

    if len(result) != before_whitespace:
        cleaning_steps.append("Cleaned excessive whitespace")

    # 3. Remove only obvious technical artifacts (very conservative)
    before_artifacts = len(result)

    # Remove null characters and other control characters (except newlines, tabs, carriage returns)
    result = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', result)

    # Remove Windows-style line endings artifacts
    result = result.replace('\r\n', '\n')
    result = result.replace('\r', '\n')

    if len(result) != before_artifacts:
        cleaning_steps.append("Removed control characters and line ending artifacts")

    # 4. Smart cleaning of leading characters using helper functions
    before_line_clean = len(result)
    lines = result.split('\n')
    cleaned_lines = []
    lines_modified = 0

    for line in lines:
        original_line = line
        cleaned_line = clean_leading_characters(line)

        if cleaned_line != original_line:
            lines_modified += 1

        # Don't add completely empty lines at the beginning
        if not cleaned_line.strip() and not cleaned_lines:
            continue

        cleaned_lines.append(cleaned_line)

    result = '\n'.join(cleaned_lines)

    if len(result) != before_line_clean:
        cleaning_steps.append(
            f"Removed ALL leading spaces and symbols - {lines_modified} lines modified (preserved structured lists flush left)")

    # 5. Very conservative HTML tag removal (only obvious tags)
    before_html = len(result)
    # Only remove clearly non-content HTML tags, preserve anything that might be content
    result = re.sub(r'<(?:br|BR)/?>', '\n', result)  # Convert <br> to newlines
    result = re.sub(r'<(?:p|P)/?>', '\n', result)  # Convert <p> to newlines
    result = re.sub(r'<(?:div|DIV)/?>', '\n', result)  # Convert <div> to newlines

    # Remove only these specific safe-to-remove tags
    safe_tags = ['html', 'head', 'body', 'meta', 'title', 'style', 'script']
    for tag in safe_tags:
        result = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', result, flags=re.IGNORECASE | re.DOTALL)
        result = re.sub(f'<{tag}[^>]*/?>', '', result, flags=re.IGNORECASE)

    if len(result) != before_html:
        cleaning_steps.append("Removed safe HTML tags")

    # 6. Clean up encoding artifacts that are clearly not content
    before_encoding = len(result)

    # Remove Unicode replacement characters
    result = result.replace('\ufffd', '')

    # Fix common encoding issues
    result = result.replace('â€™', "'")  # Smart apostrophe
    result = result.replace('â€œ', '"')  # Smart quote open
    result = result.replace('â€', '"')  # Smart quote close
    result = result.replace('â€"', '–')  # En dash
    result = result.replace('â€"', '—')  # Em dash

    if len(result) != before_encoding:
        cleaning_steps.append("Fixed encoding artifacts")

    # 7. Additional aggressive symbol cleaning
    before_symbol_cleaning = len(result)

    # Remove lines that are mostly or entirely symbols
    lines = result.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip lines that are just symbols or mostly symbols
        if stripped:
            # Count alphanumeric vs symbol characters
            alnum_count = sum(c.isalnum() or c.isspace() for c in stripped)
            symbol_count = len(stripped) - alnum_count

            # If line is more than 70% symbols, likely an artifact
            if len(stripped) > 2 and (symbol_count / len(stripped)) > 0.7:
                continue  # Skip this line

            # Skip lines that are just repeated symbols
            if re.match(r'^[=\-_~*#.>|+]{2,}$', stripped):
                continue  # Skip symbol-only lines

        filtered_lines.append(line)

    result = '\n'.join(filtered_lines)

    # Remove email reply quote markers throughout the text
    result = re.sub(r'^[\s]*>+[\s]*', '', result, flags=re.MULTILINE)

    # Remove other common email artifacts
    result = re.sub(r'^\s*[-*•·▪▫◦‣⁃]\s*$', '', result, flags=re.MULTILINE)  # Empty bullet lines

    if len(result) != before_symbol_cleaning:
        cleaning_steps.append("Aggressive symbol and email artifact removal")

    # 8. Final cleanup - ensure NO leading spaces anywhere
    before_final_spaces = len(result)

    # Split by lines and remove ALL leading whitespace from every line
    lines = result.split('\n')
    no_leading_space_lines = []

    for line in lines:
        # Remove ALL leading whitespace from every single line
        cleaned_line = line.lstrip()
        no_leading_space_lines.append(cleaned_line)

    result = '\n'.join(no_leading_space_lines)

    if len(result) != before_final_spaces:
        cleaning_steps.append("Final removal of ALL leading spaces from every line")

    # 9. Final conservative cleanup - remove only truly excessive characters
    before_final = len(result)

    # Remove multiple consecutive periods only if more than 4
    result = re.sub(r'\.{5,}', '...', result)

    # Remove multiple consecutive dashes only if more than 5
    result = re.sub(r'-{6,}', '-----', result)

    # Remove multiple consecutive equals only if more than 5
    result = re.sub(r'={6,}', '=====', result)

    if len(result) != before_final:
        cleaning_steps.append("Cleaned excessive punctuation")

    # Final trim
    result = result.strip()

    final_length = len(result)
    reduction_percent = round(((original_length - final_length) / original_length) * 100,
                              2) if original_length > 0 else 0

    return result, cleaning_steps, original_length, final_length, reduction_percent


def clean_enron_emails(json_file_path, output_file_path):
    """Main function to clean Enron emails by removing symbols and ALL leading spaces."""

    logging.info(f"Starting aggressive symbol cleaning process")
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
    skipped_emails = []
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

            logging.info("Email passed validation (has subject or body) - proceeding with symbol cleaning")

            # Create a copy of the original email to preserve all fields
            cleaned_email = email.copy()

            # Clean the email body aggressively
            original_body = email.get('Body', '')
            original_body_length = len(original_body)

            logging.info(f"Original body length: {original_body_length} characters")

            # Perform aggressive cleaning
            cleaned_body, cleaning_steps, orig_len, final_len, reduction_percent = clean_email_body(original_body)

            # Log cleaning steps
            if cleaning_steps:
                logging.info("Aggressive cleaning steps performed:")
                for step in cleaning_steps:
                    logging.info(f"  - {step}")
            else:
                logging.info("No cleaning steps needed")

            logging.info(f"Body length after cleaning: {final_len} characters")
            logging.info(f"Size reduction: {reduction_percent}%")

            # Update the Body field with cleaned content
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
    logging.info("SYMBOL CLEANING PROCESS SUMMARY")
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
    logging.info("Aggressive symbol cleaning process completed!")

    return cleaned_emails


# Usage
if __name__ == "__main__":
    input_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron.json"
    output_file = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_enron_emails_aggressive.json"

    print("Starting aggressive symbol cleaning for Enron emails...")
    print("This version removes unnecessary symbols and ALL leading spaces")
    print("Check 'email_cleaning.log' for detailed logs")

    cleaned_data = clean_enron_emails(input_file, output_file)

    print(f"\nProcess completed! Check the log file for details.")
    print(f"Processed {len(cleaned_data)} emails successfully with aggressive symbol cleaning.")
    print("Check 'cleaned_enron_emails_aggressive_skipped_report.json' for details on skipped emails.")