import sys
import os
import json
import spacy

# Add src to sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.config import config

# ----- File paths from config -----
INPUT_PATH = config.CLEANED_JSON_PATH
OUTPUT_PATH = config.EXTRACTED_ENTITIES_JSON_PATH

# ----- Load spaCy model -----
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# ----- Load input emails -----
print(f"Loading emails from: {INPUT_PATH}")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    emails = json.load(f)

output = []
total = len(emails)
print(f"Processing {total} emails...")

# ----- Process emails -----
for idx, email in enumerate(emails, 1):
    # Extract fields
    message_id = email.get("Message-ID", "")
    sender = email.get("From", "")
    to = email.get("To", "")
    subject = email.get("Subject", "")
    body = email.get("Body", "")
    source_file = email.get("SourceFile", "")

    # spaCy NER
    combined_text = f"{subject}\n{body}"
    doc = nlp(combined_text)
    entities = [{"value": ent.text, "label": ent.label_} for ent in doc.ents]

    # Build output dict with SourceFile as last key
    result = {
        "Message-ID": message_id,
        "From": sender,
        "To": to,
        "entities": entities,
        "SourceFile": source_file
    }
    output.append(result)

    # Console progress message every 100 emails and at end
    if idx % 100 == 0 or idx == total:
        print(f"Processed {idx} / {total} emails.")

# ----- Write output -----
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Done! Extracted entities written to {OUTPUT_PATH}")
