import sys
import os
import json
import spacy
import re

from typing import List, Dict

# Add project root to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.config import config

INPUT_PATH = config.CLEANED_JSON_PATH
OUTPUT_PATH = config.EXTRACTED_ENTITIES_JSON_PATH

# A small utility to clean whitespace and stray punctuation
def clean_text(text: str) -> str:
    text = text.strip()
    # collapse multiple whitespace/newlines/tabs
    text = re.sub(r'\s+', ' ', text)
    # remove trailing colons, commas, periods
    text = re.sub(r'[,:\.]+$', '', text)
    return text

# A simple validator to drop obviously junk entities
def is_valid_entity(ent_text: str, ent_label: str) -> bool:
    ent_text = ent_text.strip()
    # drop anything shorter than 2 characters
    if len(ent_text) < 2:
        return False
    # drop if it’s purely punctuation/digits
    if re.fullmatch(r'[\W\d]+', ent_text):
        return False
    # for PERSON: require at least one space or capital letter
    if ent_label == "PERSON":
        if not re.search(r"[A-Za-z]", ent_text):
            return False
    # no further special rules—trust the transformer
    return True

# Remove duplicates while preserving order
def dedupe_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []
    for ent in entities:
        key = (ent["value"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique

# 1) Load spaCy transformer model
print("Loading spaCy transformer model en_core_web_trf ...")
try:
    nlp = spacy.load("en_core_web_trf")
    print("✓ Loaded en_core_web_trf (transformer-based) successfully")
except OSError:
    print("❌ en_core_web_trf not installed. Falling back to en_core_web_lg ...")
    try:
        nlp = spacy.load("en_core_web_lg")
        print("✓ Loaded en_core_web_lg (fallback)")
    except OSError:
        print("❌ en_core_web_lg not found. Please run:")
        print("    python -m spacy download en_core_web_trf")
        sys.exit(1)

# 2) Load input JSON (list of emails)
print(f"Loading emails from: {INPUT_PATH}")
try:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)
except Exception as e:
    print(f"❌ Failed to load input JSON: {e}")
    sys.exit(1)

output = []
total_emails = len(emails)
print(f"Processing {total_emails} emails ...")

# 3) Iterate and extract entities
for idx, email in enumerate(emails, 1):
    message_id = email.get("Message-ID", "")
    sender = email.get("From", "")
    recipient = email.get("To", "")
    subject = email.get("Subject", "")
    body = email.get("Body", "")
    source_file = email.get("SourceFile", "")

    text_to_process = "\n".join(filter(None, [subject, body]))
    ents_cleaned: List[Dict[str, str]] = []

    if text_to_process.strip():
        doc = nlp(text_to_process)
        for ent in doc.ents:
            # Clean punctuation/whitespace
            cleaned = clean_text(ent.text)
            if not cleaned:
                continue
            # Simple label filter (only PERSON, ORG, GPE, maybe DATE/WORK_OF_ART)
            if ent.label_ not in {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART"}:
                continue
            if not is_valid_entity(cleaned, ent.label_):
                continue
            ents_cleaned.append({"value": cleaned, "label": ent.label_})

    # Deduplicate
    ents_cleaned = dedupe_entities(ents_cleaned)

    output.append({
        "Message-ID": message_id,
        "From": sender,
        "To": recipient,
        "entities": ents_cleaned,
        "SourceFile": source_file
    })

    if idx % 100 == 0 or idx == total_emails:
        print(f"  Processed {idx}/{total_emails}")

# 4) Write out to JSON
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote extracted entities to: {OUTPUT_PATH}")
except Exception as e:
    print(f"❌ Error writing output: {e}")
    sys.exit(1)

print("Done.")
