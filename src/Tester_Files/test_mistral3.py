#!/usr/bin/env python3
"""
Single-file Email Analyzer using Ollama Mistral
Configure the input and output file paths in the CONFIGURATION section below, then run: python email_analyzer.py
"""

import json
import re
import ollama
from typing import Dict, List, Any

# ==================== CONFIGURATION ====================
INPUT_FILE = "D:\Projects\HMI\HMI_Project\data\cleaned_body_chain_enron.json"  # Change this to your input file path
OUTPUT_FILE = "D:\Projects\HMI\HMI_Project\data\enron_analysis_results.json"  # Change this to your desired output file path
MAX_EMAILS = 50  # Process only first N emails
BATCH_SIZE = 10  # Process emails in batches of N
# ========================================================

def extract_email_data(raw_content: str) -> List[Dict[str, Any]]:
    """Extract email data from raw JSON content"""
    emails = []

    # Try to parse as JSON first
    try:
        data = json.loads(raw_content)
        if isinstance(data, list):
            # If it's already a list of emails, extend with it
            emails.extend(data)
        else:
            # If it's a single email object, append it
            emails.append(data)
        return emails
    except json.JSONDecodeError:
        pass

    # Handle multiple emails or malformed JSON using regex
    email_patterns = {
        'from': r'"From":\s*"([^"]+)"',
        'to': r'"To":\s*"([^"]+)"',
        'subject': r'"Subject":\s*"([^"]+)"',
        'date': r'"Date":\s*"([^"]+)"',
        'body': r'"Body":\s*"([^"]*(?:\\.[^"]*)*)"'
    }

    # Find all Message-ID positions to split emails
    message_ids = list(re.finditer(r'"Message-ID":\s*"([^"]+)"', raw_content))

    for i, match in enumerate(message_ids):
        start_pos = match.start()
        end_pos = message_ids[i + 1].start() if i + 1 < len(message_ids) else len(raw_content)
        email_section = raw_content[start_pos:end_pos]

        email_data = {}
        for field, pattern in email_patterns.items():
            match = re.search(pattern, email_section)
            if match:
                value = match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                email_data[field] = value

        # Extract BodyChain content if present
        bodychain_match = re.search(r'"BodyChain":\s*\[.*?"Body":\s*"([^"]*(?:\\.[^"]*)*)"', email_section, re.DOTALL)
        if bodychain_match:
            bodychain_content = bodychain_match.group(1).replace('\\n', '\n').replace('\\"', '"')
            email_data['bodychain_content'] = bodychain_content

        emails.append(email_data)

    return emails


def query_mistral(prompt: str, model_name: str = "mistral") -> str:
    """Send prompt to Mistral via Ollama"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error querying Mistral: {e}")
        return ""


def analyze_summary(email_content: str) -> str:
    """Generate summary using Mistral"""
    prompt = f"""
    Provide a concise 2-3 sentence summary of this email content. Focus on the main points and key information:

    Email Content:
    {email_content[:2000]}

    Summary:
    """
    return query_mistral(prompt)


def analyze_tone(email_content: str) -> str:
    """Analyze tone using Mistral"""
    prompt = f"""
    Analyze the tone of this email and classify it as one of these categories:
    - Urgent/Alerting
    - Collaborative/Professional  
    - Concerned/Critical
    - Formal/Regulatory
    - Neutral/Informational

    Email Content:
    {email_content[:1500]}

    Respond with just the tone category:
    """
    return query_mistral(prompt)


def classify_email(subject: str, content: str) -> str:
    """Classify email type using Mistral"""
    prompt = f"""
    Based on the subject and content, classify this email into one of these categories:
    - Regulatory Alert / Crisis Communication
    - Business Project Coordination
    - Legal/Compliance Matter
    - Market/Trading Information
    - Internal Communication
    - Strategic Planning

    Subject: {subject}
    Content: {content[:1000]}

    Respond with just the classification:
    """
    return query_mistral(prompt)


def extract_entities(email_content: str) -> Dict[str, List[str]]:
    """Extract entities using Mistral with fallback"""
    prompt = f"""
    Extract entities from this email and return ONLY a valid JSON object with these exact keys:
    {{"people": [], "organizations": [], "locations": [], "dates": [], "projects": [], "topics": []}}

    - people: Names and email addresses
    - organizations: Company names, agencies
    - locations: Cities, states, countries
    - dates: Any dates mentioned
    - projects: Project names, bills, initiatives
    - topics: Key business/technical topics

    Email Content:
    {email_content[:2000]}

    JSON:
    """

    response = query_mistral(prompt)

    # Try to parse JSON response
    try:
        # Clean response to extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = response[json_start:json_end]
            entities = json.loads(json_str)
            return entities
    except:
        pass

    # Fallback: extract using regex
    return fallback_entity_extraction(email_content)


def fallback_entity_extraction(text: str) -> Dict[str, List[str]]:
    """Fallback entity extraction using regex"""
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "projects": [],
        "topics": []
    }

    # Email addresses
    entities["people"].extend(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))

    # Names
    entities["people"].extend(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text))

    # Organizations
    org_pattern = r'\b(Enron|FERC|Federal Energy Regulatory Commission|California|PUC|El Paso|Merchant Energy|Socalgas|ECT|Dynegy|Williams|Duke Energy|Calpine)\b'
    entities["organizations"].extend(re.findall(org_pattern, text, re.IGNORECASE))

    # Locations
    loc_pattern = r'\b(California|Washington|Arizona|Alaska|Tennessee|Pacific Northwest|Western states)\b'
    entities["locations"].extend(re.findall(loc_pattern, text, re.IGNORECASE))

    # Dates
    entities["dates"].extend(re.findall(r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b|\b\w+ \d{1,2}, \d{4}\b', text))

    # Projects
    entities["projects"].extend(re.findall(r'\b(Project Boomerang|AB1890)\b', text, re.IGNORECASE))

    # Topics
    topic_words = ['electricity', 'overcharges', 'refund', 'power', 'energy', 'price', 'gas', 'deregulation', 'trading',
                   'crisis', 'regulation']
    entities["topics"] = [word for word in topic_words if word.lower() in text.lower()]

    # Remove duplicates and limit results
    for key in entities:
        entities[key] = list(set(entities[key]))[:10]

    return entities


def analyze_email(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single email and return structured results"""
    # Handle different field name formats (From/from, To/to, etc.)
    email_from = email_data.get('From') or email_data.get('from', '')
    email_to = email_data.get('To') or email_data.get('to', '')
    email_subject = email_data.get('Subject') or email_data.get('subject', '')
    email_date = email_data.get('Date') or email_data.get('date', '')
    email_body = email_data.get('Body') or email_data.get('body', '')

    # Handle BodyChain content
    bodychain_content = ''
    if 'BodyChain' in email_data and email_data['BodyChain']:
        # Extract body content from BodyChain array
        for chain_item in email_data['BodyChain']:
            if isinstance(chain_item, dict) and 'Body' in chain_item:
                bodychain_content += chain_item['Body'] + '\n\n'
    elif 'bodychain_content' in email_data:
        bodychain_content = email_data.get('bodychain_content', '')

    # Combine body and bodychain content
    full_content = email_body + '\n\n' + bodychain_content

    print(f"  Generating summary...")
    summary = analyze_summary(full_content)

    print(f"  Analyzing tone...")
    tone = analyze_tone(full_content)

    print(f"  Classifying email...")
    classification = classify_email(email_subject, full_content)

    print(f"  Extracting entities...")
    entities = extract_entities(full_content)

    return {
        "to": email_to,
        "from": email_from,
        "date": email_date,
        "subject": email_subject,
        "summary": summary,
        "tone_analysis": tone,
        "classification": classification,
        "entities": entities
    }


def main():
    """Main function"""
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Test Ollama connection
    print("Testing Ollama Mistral connection...")
    try:
        test_response = query_mistral("Respond with 'OK' if you can see this.")
        if not test_response or 'ok' not in test_response.lower():
            print("❌ Could not connect to Ollama Mistral")
            print("Make sure to:")
            print("1. Install Ollama: https://ollama.ai/")
            print("2. Pull Mistral: ollama pull mistral")
            print("3. Start Ollama: ollama serve")
            return
        print("✓ Connected to Ollama Mistral")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return

    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        print(f"✓ Read input file: {input_file}")
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Extract email data
    emails = extract_email_data(raw_content)
    if not emails:
        print("❌ No emails found in the file")
        return

    # Limit to first MAX_EMAILS
    if len(emails) > MAX_EMAILS:
        emails = emails[:MAX_EMAILS]
        print(f"✓ Limited to first {MAX_EMAILS} emails (out of {len(extract_email_data(raw_content))} total)")

    print(f"✓ Processing {len(emails)} email(s) in batches of {BATCH_SIZE}")

    # Analyze emails in batches
    results = {}
    total_emails = len(emails)

    for batch_start in range(0, total_emails, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_emails)
        batch_emails = emails[batch_start:batch_end]

        print(f"\n--- Processing Batch {batch_start // BATCH_SIZE + 1} (emails {batch_start + 1}-{batch_end}) ---")

        for i, email_data in enumerate(batch_emails):
            email_num = batch_start + i + 1
            print(f"\nAnalyzing email {email_num}...")
            results[f"email_{email_num}"] = analyze_email(email_data)
            print(f"✓ Email {email_num} analysis complete")

        print(f"✓ Batch {batch_start // BATCH_SIZE + 1} completed ({len(batch_emails)} emails)")

    print(f"\n✓ All {total_emails} emails processed successfully!")

    # Write output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Analysis complete! Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        return

    # Show summary
    print(f"\nSummary ({len(results)} emails processed):")
    for email_id, result in results.items():
        print(f"  {email_id}: {result['classification']} - {result['tone_analysis']}")


if __name__ == "__main__":
    main()
