import sys
import os
import json
import re

from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.config import config

INPUT_PATH = config.CLEANED_JSON_PATH
OUTPUT_PATH = config.EXTRACTED_ENTITIES_JSON_PATH

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,:\.]+$', '', text)
    return text

def is_valid_entity(ent_text: str, ent_label: str) -> bool:
    ent_text = ent_text.strip()
    if len(ent_text) < 2:
        return False
    if re.fullmatch(r'[\W\d]+', ent_text):
        return False
    # If your BERT-NER uses "PER" instead of "PERSON", adjust here:
    if ent_label == "PER":
        if not re.search(r"[A-Za-z]", ent_text):
            return False
    return True

def dedupe_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []
    for ent in entities:
        key = (ent["value"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique

from flair.data import Sentence
from flair.models import SequenceTagger

MODEL_PATH = config.MODEL_PATH
print(f"Loading Flair SequenceTagger from: {MODEL_PATH}")
try:
    tagger = SequenceTagger.load(MODEL_PATH)
    print("✓ Successfully loaded BERT-NER SequenceTagger")
except Exception as e:
    print(f"❌ Error loading Flair tagger at {MODEL_PATH}: {e}")
    sys.exit(1)

print(f"Loading emails from: {INPUT_PATH}")
try:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)
except Exception as e:
    print(f"❌ Failed to load input JSON: {e}")
    sys.exit(1)

output = []
total_emails = len(emails)
print(f"Processing {total_emails} emails …")

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
        sentence = Sentence(text_to_process)
        tagger.predict(sentence)

        for span in sentence.get_spans('ner'):
            ent_text = span.text
            ent_label = span.labels[0].value   # e.g. "PER","ORG","LOC","DATE","MISC"
            ent_score = span.labels[0].score
            cleaned = clean_text(ent_text)
            if not cleaned:
                continue

            # Filter to exactly the labels you care about:
            if ent_label not in {"PER", "ORG", "LOC", "DATE", "MISC"}:
                continue

            if not is_valid_entity(cleaned, ent_label):
                continue

            ents_cleaned.append({
                "value":      cleaned,
                "label":      ent_label,
                "confidence": ent_score
            })

    ents_cleaned = dedupe_entities(ents_cleaned)

    output.append({
        "Message-ID": message_id,
        "From": sender,
        "To": recipient,
        "entities": ents_cleaned,
        "SourceFile": source_file
    })

    if idx % 100 == 0 or idx == total_emails:
        print(f"  Processed {idx}/{total_emails} emails")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
try:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote extracted entities to: {OUTPUT_PATH}")
except Exception as e:
    print(f"❌ Error writing output: {e}")
    sys.exit(1)

print("Done.")
