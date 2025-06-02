import sys
import os
import json
import spacy
import re
from typing import List, Dict, Any

# Add src to sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.config import config

# ----- File paths from config -----
INPUT_PATH = config.CLEANED_JSON_PATH
OUTPUT_PATH = config.EXTRACTED_ENTITIES_JSON_PATH


# ----- Entity cleaning and validation functions -----

def clean_entity_text(text: str) -> str:
    """Clean entity text by removing unwanted characters and whitespace"""
    # Remove extra whitespace, newlines, and tabs
    cleaned = re.sub(r'\s+', ' ', text.strip())

    # Remove common email artifacts
    cleaned = re.sub(r'@\w+\s*', '', cleaned)  # Remove @domain parts
    cleaned = re.sub(r'\n+', ' ', cleaned)  # Replace newlines with space
    cleaned = re.sub(r'\t+', ' ', cleaned)  # Replace tabs with space

    # Remove trailing punctuation and common suffixes
    cleaned = re.sub(r'[:\n\t\r]+$', '', cleaned)  # Remove trailing colons, newlines
    cleaned = re.sub(r'\s*Subject\s*$', '', cleaned, re.IGNORECASE)  # Remove "Subject"
    cleaned = re.sub(r'\s*Re:\s*$', '', cleaned, re.IGNORECASE)  # Remove "Re:"
    cleaned = re.sub(r'\s*Fw:\s*$', '', cleaned, re.IGNORECASE)  # Remove "Fw:"

    # Remove common prefixes that spaCy might include
    cleaned = re.sub(r'^(Dear|Hi|Hello)\s+', '', cleaned, re.IGNORECASE)

    # Remove trailing dots and commas
    cleaned = re.sub(r'[.,]+$', '', cleaned)

    return cleaned.strip()


def is_valid_person(text: str) -> bool:
    """Validate if text is likely a person name"""
    text = text.strip()

    # Too short or too long
    if len(text) < 2 or len(text) > 50:
        return False

    # Common false positives
    invalid_patterns = [
        r'^(thanks?|regards?|best|sincerely|cheers?)$',  # Common closings
        r'^(re|fw|fwd)[:,\s]*$',  # Email prefixes
        r'^(subject|from|to|cc|bcc)[:,\s]*$',  # Email headers
        r'^[^a-zA-Z]*$',  # No letters
        r'^\s*(hi|hello|dear)\s*$',  # Greetings alone
        r'^\d+$',  # Only numbers
        r'^[^\w\s]+$',  # Only punctuation
        r'^(meeting|call|phone|email)s?$',  # Common words
        r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',  # Days
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',  # Months
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    # Should contain at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False

    # Should not be all caps with too many non-letter characters
    if text.isupper() and len(re.sub(r'[a-zA-Z\s]', '', text)) > len(text) // 2:
        return False

    # Better name validation - should have at least one proper word
    words = text.split()
    if len(words) > 0 and all(len(word) < 2 for word in words):
        return False

    return True


def is_valid_organization(text: str) -> bool:
    """Validate if text is likely an organization"""
    text = text.strip()

    # Too short or too long
    if len(text) < 2 or len(text) > 100:
        return False

    # Common false positives for organizations
    invalid_patterns = [
        r'^(thanks?|regards?|best|sincerely|cheers?)$',  # Common closings
        r'^(re|fw|fwd)[:,\s]*$',  # Email prefixes
        r'^(subject|from|to|cc|bcc)[:,\s]*$',  # Email headers
        r'^(hi|hello|dear)$',  # Greetings
        r'^[^a-zA-Z]*$',  # No letters
        r'^\d+$',  # Only numbers
        r'^[^\w\s]+$',  # Only punctuation
        r'^(meeting|call|phone|email)s?$',  # Common words
        r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',  # Days
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)$',  # Months
        r'^(morning|afternoon|evening|night|today|tomorrow|yesterday)$',  # Time references
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    # Should contain at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False

    return True


def is_valid_location(text: str) -> bool:
    """Validate if text is likely a location (GPE - Geopolitical Entity)"""
    text = text.strip()

    # Too short or too long
    if len(text) < 2 or len(text) > 50:
        return False

    # Common false positives
    invalid_patterns = [
        r'^(thanks?|regards?|best|sincerely)$',  # Common closings
        r'^[^a-zA-Z]*$',  # No letters
        r'^\d+$',  # Only numbers
        r'^[^\w\s]+$',  # Only punctuation
        r'^(meeting|call|phone|email)s?$',  # Common words
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    return True


def is_valid_date(text: str) -> bool:
    """Validate if text is likely a valid date"""
    text = text.strip()

    # Should not be too short
    if len(text) < 3:
        return False

    # Common invalid date patterns
    invalid_patterns = [
        r'^[^\d/\-\w\s]+$',  # Only punctuation (no numbers, letters, slashes, dashes)
        r'^(re|fw|fwd)$',  # Email prefixes
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    return True


def is_valid_entity(text: str, label: str) -> bool:
    """Main validation function for entities"""
    text = text.strip()

    # General checks
    if not text or len(text) < 2:
        return False

    # Label-specific validation
    if label == "PERSON":
        return is_valid_person(text)
    elif label == "ORG":
        return is_valid_organization(text)
    elif label == "GPE":  # Geopolitical entities (countries, cities, states)
        return is_valid_location(text)
    elif label == "DATE":
        return is_valid_date(text)
    elif label in ["TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"]:
        # These are usually more reliable from spaCy
        return len(text.strip()) > 0 and not re.match(r'^[^\w\d]+$', text)
    else:
        # For other labels, basic validation
        return len(text.strip()) > 1 and re.search(r'[a-zA-Z]', text)


def extract_and_clean_entities(text: str, nlp) -> List[Dict[str, str]]:
    """Extract entities and clean/validate them"""
    if not text or not text.strip():
        return []

    # Split very long texts to avoid processing issues
    max_length = 1000000  # 1MB character limit for spaCy
    if len(text) > max_length:
        # Process in chunks and combine results
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length - 1000)]
        all_entities = []
        for chunk in chunks:
            chunk_entities = extract_and_clean_entities(chunk, nlp)
            all_entities.extend(chunk_entities)
        return remove_duplicates(all_entities)

    try:
        doc = nlp(text)
    except Exception as e:
        print(f"Error processing text with spaCy: {e}")
        return []

    entities = []

    for ent in doc.ents:
        # Clean the entity text
        cleaned_text = clean_entity_text(ent.text)

        # Skip if cleaning resulted in empty text
        if not cleaned_text:
            continue

        # Validate the entity
        if is_valid_entity(cleaned_text, ent.label_):
            entities.append({
                "value": cleaned_text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

    return remove_duplicates(entities)


def remove_duplicates(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate entities while preserving order"""
    seen = set()
    unique_entities = []

    for entity in entities:
        # Create a key for comparison (case-insensitive value + label)
        entity_key = (entity["value"].lower(), entity["label"])
        if entity_key not in seen:
            seen.add(entity_key)
            # Remove position info for final output
            unique_entities.append({
                "value": entity["value"],
                "label": entity["label"]
            })

    return unique_entities


# ----- Load spaCy model -----
print("Loading spaCy large model (en_core_web_lg)...")
try:
    nlp = spacy.load("en_core_web_lg")
    print("✓ Successfully loaded en_core_web_lg")
except OSError:
    print("❌ en_core_web_lg not found. Please install it with:")
    print("python -m spacy download en_core_web_lg")
    print("\nFalling back to en_core_web_sm...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ Successfully loaded en_core_web_sm (fallback)")
    except OSError:
        print("❌ No spaCy models found. Please install at least en_core_web_sm:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

# ----- Load input emails -----
print(f"Loading emails from: {INPUT_PATH}")
try:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)
except FileNotFoundError:
    print(f"❌ Input file not found: {INPUT_PATH}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"❌ Invalid JSON in input file: {INPUT_PATH}")
    sys.exit(1)

output = []
total = len(emails)
print(f"Processing {total} emails...")

# Statistics tracking
stats = {
    "total_emails": total,
    "emails_with_entities": 0,
    "total_entities_found": 0,
    "total_entities_after_cleaning": 0,
    "entity_types": {},
    "processing_errors": 0
}

# ----- Process emails -----
for idx, email in enumerate(emails, 1):
    try:
        # Extract fields with safe defaults
        message_id = email.get("Message-ID", "")
        sender = email.get("From", "")
        to = email.get("To", "")
        subject = email.get("Subject", "")
        body = email.get("Body", "")
        source_file = email.get("SourceFile", "")

        # Extract and clean entities
        combined_text = f"{subject}\n{body}" if subject or body else ""
        entities = extract_and_clean_entities(combined_text, nlp)

        # Update statistics
        if entities:
            stats["emails_with_entities"] += 1
            stats["total_entities_after_cleaning"] += len(entities)

            for entity in entities:
                label = entity["label"]
                stats["entity_types"][label] = stats["entity_types"].get(label, 0) + 1

        # Build output dict with SourceFile as last key
        result = {
            "Message-ID": message_id,
            "From": sender,
            "To": to,
            "entities": entities,
            "SourceFile": source_file
        }
        output.append(result)

    except Exception as e:
        print(f"Error processing email {idx}: {e}")
        stats["processing_errors"] += 1
        # Add empty result to maintain order
        result = {
            "Message-ID": email.get("Message-ID", ""),
            "From": email.get("From", ""),
            "To": email.get("To", ""),
            "entities": [],
            "SourceFile": email.get("SourceFile", "")
        }
        output.append(result)

    # Console progress message every 100 emails and at end
    if idx % 100 == 0 or idx == total:
        print(f"Processed {idx} / {total} emails.")

# ----- Write output -----
try:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Output written to: {OUTPUT_PATH}")
except Exception as e:
    print(f"❌ Error writing output file: {e}")
    sys.exit(1)

# ----- Print statistics -----
print(f"\n=== PROCESSING COMPLETE ===")
print(f"Total emails processed: {stats['total_emails']}")
print(f"Emails with entities: {stats['emails_with_entities']}")
print(f"Processing errors: {stats['processing_errors']}")
print(f"Total entities found after cleaning: {stats['total_entities_after_cleaning']}")

if stats['total_entities_after_cleaning'] > 0:
    print(f"\nEntity types distribution:")
    for entity_type, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_entities_after_cleaning']) * 100
        print(f"  {entity_type}: {count} ({percentage:.1f}%)")

print(f"\nDone! Cleaned entities written to {OUTPUT_PATH}")

# ----- Installation reminder -----
if 'en_core_web_sm' in str(nlp.meta):
    print(f"\n💡 TIP: For better accuracy, install the large model:")
    print(f"python -m spacy download en_core_web_lg")