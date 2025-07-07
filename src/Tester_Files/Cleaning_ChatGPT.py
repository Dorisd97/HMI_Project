# import json
# import re
# from src.config.config import CLEANED_JSON_PATH
#
#
# def clean_email_body(body):
#     if not body:
#         return ""
#
#     # Remove HTML/XML tags
#     body = re.sub(r'<[^>]+>', '', body)
#     # Remove common encoding artifacts
#     body = re.sub(r'=20|=09|=0A|=0D|=2E', ' ', body)
#     body = re.sub(r'\\n|\\r|\\t', ' ', body)
#     # Remove quoted-printable encoding leftovers
#     body = re.sub(r'=[a-zA-Z0-9]{2}', '', body)
#     # Remove namespace or XML artifacts
#     body = re.sub(r'<?xml.*?>', '', body)
#     # Remove any repeated dashes, underscores, or equals (common signature dividers)
#     body = re.sub(r'[-_=]{5,}', '', body)
#     # Remove email footers or legal disclaimers by common patterns
#     body = re.sub(r'This e-mail.*?(\n|\Z)', '', body, flags=re.I | re.S)
#     body = re.sub(r'Forwarded by.*?(\n|\Z)', '', body, flags=re.I | re.S)
#     # Remove excess reply chains (keep only last 2, for example)
#     parts = re.split(r'\n\s*(From:|On .* wrote:|-----Original Message-----)', body)
#     if len(parts) > 4:
#         body = ''.join(parts[:4])
#     # Normalize whitespace
#     body = re.sub(r'\s+', ' ', body)
#     # Trim leading/trailing whitespace
#     return body.strip()
#
#
# def process_json_file(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     for email in data:
#         original = email.get('Body', '')
#         cleaned = clean_email_body(original)
#         email['Body_Cleaned'] = cleaned  # Add a new field with the cleaned body
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#
#
# # Example usage
# input_file = 'D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_json_50.json'
# output_file = 'D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_json_50_output.json'
# process_json_file(input_file, output_file)
import json
import re
import logging

# Set up logging (console and file)
logging.basicConfig(
    filename='email_cleaning_verbose.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def clean_email_body(body, msg_id):
    if not body:
        logging.info(f"{msg_id} - Empty body; nothing to clean.")
        return ""

    logging.info(f"{msg_id} - Original body: {repr(body[:80])}...")

    # 1. Remove HTML/XML tags
    new_body = re.sub(r'<[^>]+>', '', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed HTML/XML tags.")
    body = new_body

    # 2. Remove common encoding artifacts (=20, =09, etc.)
    new_body = re.sub(r'=20|=09|=0A|=0D|=2E', ' ', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed encoding artifacts (=XX).")
    body = new_body

    # 3. Remove escaped line breaks/tabs
    new_body = re.sub(r'\\n|\\r|\\t', ' ', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed escaped line breaks/tabs.")
    body = new_body

    # 4. Remove quoted-printable encoding leftovers (e.g., =3D)
    new_body = re.sub(r'=[a-zA-Z0-9]{2}', '', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed quoted-printable leftovers.")
    body = new_body

    # 5. Remove XML namespace artifacts
    new_body = re.sub(r'<?xml.*?>', '', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed XML namespace artifacts.")
    body = new_body

    # 6. Remove repeated dashes, underscores, or equals (signature dividers)
    new_body = re.sub(r'[-_=]{5,}', '', body)
    if new_body != body:
        logging.info(f"{msg_id} - Removed repeated signature dividers.")
    body = new_body

    # 7. Remove legal disclaimers/footers (patterns)
    new_body = re.sub(r'This e-mail.*?(\n|\Z)', '', body, flags=re.I|re.S)
    if new_body != body:
        logging.info(f"{msg_id} - Removed 'This e-mail...' disclaimer/footer.")
    body = new_body

    new_body = re.sub(r'Forwarded by.*?(\n|\Z)', '', body, flags=re.I|re.S)
    if new_body != body:
        logging.info(f"{msg_id} - Removed 'Forwarded by...' disclaimer/footer.")
    body = new_body

    # 8. Remove excess reply chains (keep only last 2, for example)
    parts = re.split(r'\n\s*(From:|On .* wrote:|-----Original Message-----)', body)
    if len(parts) > 4:
        logging.info(f"{msg_id} - Truncated reply chain (original parts: {len(parts)}) to 2 sections.")
        body = ''.join(parts[:4])
    else:
        logging.info(f"{msg_id} - Reply chain within acceptable length.")

    # 9. Normalize whitespace
    new_body = re.sub(r'\s+', ' ', body)
    if new_body != body:
        logging.info(f"{msg_id} - Normalized whitespace.")
    body = new_body

    # 10. Strip leading/trailing whitespace
    body = body.strip()
    logging.info(f"{msg_id} - Final cleaned body: {repr(body[:80])}...")

    return body

def process_json_file(input_file, output_file):
    logging.info('Starting email cleaning process.')
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f'Error reading input file: {e}')
        return

    processed_count = 0
    for email in data:
        msg_id = email.get('Message-ID', '[NO ID]')
        logging.info(f"Processing email: {msg_id}")
        original = email.get('Body', '')
        cleaned = clean_email_body(original, msg_id)
        email['Body_Cleaned'] = cleaned

        if original.strip() != cleaned.strip():
            logging.info(f"{msg_id} - Changes detected and applied.")
        else:
            logging.info(f"{msg_id} - No changes needed.")

        processed_count += 1

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f'Email cleaning completed. {processed_count} emails processed.')
        logging.info(f'Output written to {output_file}')
    except Exception as e:
        logging.error(f'Error writing output file: {e}')

# Example usage
input_file = 'D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_json_50.json'
output_file = 'D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/cleaned_json_50_Output.json'
process_json_file(input_file, output_file)
