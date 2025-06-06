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
OUTPUT_FILE = "D:\Projects\HMI\HMI_Project\data\enron_test_analysis_results.json"  # Change this to your desired output file path
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
    {{"people": [], "organizations": [], "locations": [], "dates": [], "projects": [], "legal": [], "topics": []}}

    - people: ONLY person names (not email addresses) - extract full names like "John Smith", "Mary Johnson"  
    - organizations: Company names, agencies, institutions (like "Enron Corp", "FERC", "Duke Energy")
    - locations: Cities, states, countries, regions (like "California", "Texas", "New York")
    - dates: Any dates mentioned in any format
    - projects: ONLY actual project names (like "Project Boomerang", "Project Alpha", "Whitewing Project")
    - legal: Bills, acts, sections, orders, dockets, legal references (like "AB1890", "Section 5", "FERC Order 2000")
    - topics: Key business/technical topics and themes

    Email Content:
    {email_content[:2000]}

    Return only the JSON object:
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

            # Clean and validate the response
            cleaned_entities = clean_extracted_entities(entities)
            return cleaned_entities
    except:
        pass

    # Fallback: extract using regex
    return fallback_entity_extraction(email_content)


def clean_extracted_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Clean and validate extracted entities"""
    cleaned = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "projects": [],
        "legal": [],
        "topics": []
    }

    # Clean people - remove email addresses, keep only names
    if "people" in entities:
        for person in entities["people"]:
            if isinstance(person, str):
                # Skip email addresses
                if "@" not in person and len(person.strip()) > 2:
                    # Basic name validation (at least 2 words, starts with capital)
                    words = person.strip().split()
                    if len(words) >= 2 and all(word[0].isupper() for word in words if word.isalpha()):
                        cleaned["people"].append(person.strip())

    # Clean other categories
    for category in ["organizations", "locations", "dates", "projects", "legal", "topics"]:
        if category in entities and isinstance(entities[category], list):
            for item in entities[category]:
                if isinstance(item, str) and len(item.strip()) > 1:
                    cleaned[category].append(item.strip())

    # Remove duplicates and limit
    for key in cleaned:
        cleaned[key] = list(dict.fromkeys(cleaned[key]))[:15]  # Preserve order, remove dupes

    return cleaned


def fallback_entity_extraction(text: str) -> Dict[str, List[str]]:
    """Enhanced fallback entity extraction using comprehensive regex patterns"""
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "projects": [],
        "legal": [],
        "topics": []
    }

    # PEOPLE - Extract names only (no email addresses)
    # Pattern for proper names (First Last, First Middle Last, etc.)
    name_patterns = [
        r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)?\s*([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b',  # Titles + names
        r'\b([A-Z][a-z]{2,20}\s+[A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{1,20})?)\b',  # First Last (Middle)
        r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # First M. Last
    ]

    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            name = match.strip() if isinstance(match, str) else match[0].strip()
            # Exclude common false positives
            if not any(word.lower() in ['energy', 'power', 'gas', 'electric', 'corp', 'company', 'inc', 'llc', 'ltd']
                       for word in name.split()):
                entities["people"].append(name)

    # ORGANIZATIONS - Comprehensive company and agency patterns
    org_patterns = [
        # Energy companies
        r'\b(Enron(?:\s+Corp)?|FERC|Federal Energy Regulatory Commission|PUC|Public Utilities Commission)\b',
        r'\b(El Paso|Merchant Energy|Socalgas|Southern California Gas|ECT|Dynegy|Williams|Duke Energy)\b',
        r'\b(Calpine|Reliant|TXU|Mirant|NRG|Edison|PG&E|Pacific Gas)\b',
        r'\b(Sempra|Kinder Morgan|TransCanada|Dominion|Constellation)\b',

        # Government/Regulatory
        r'\b(SEC|CFTC|DOL|EPA|DOE|Department of Energy|Securities and Exchange Commission)\b',
        r'\b(Senate|House of Representatives|Congress|CPUC|California Public Utilities Commission)\b',
        r'\b(NERC|North American Electric Reliability Council|ISO|Independent System Operator)\b',

        # Companies with suffixes
        r'\b([A-Z][A-Za-z\s&]{2,30}(?:Corp|Corporation|Inc|Incorporated|LLC|Ltd|Limited|LP|LLP|Company|Co)\b)',
        r'\b([A-Z][A-Za-z\s&]{2,30}(?:Energy|Power|Gas|Electric|Utility|Trading|Marketing|Services)\b)',

        # Financial/Legal
        r'\b(Goldman Sachs|Morgan Stanley|JPMorgan|Bank of America|Wells Fargo|Citigroup)\b',
        r'\b(Skadden|Vinson|Baker Botts|Latham|White & Case|Jones Day)\b'
    ]

    for pattern in org_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            org = match if isinstance(match, str) else match[0] if match else ""
            if org and len(org.strip()) > 2:
                entities["organizations"].append(org.strip())

    # LOCATIONS - US states, major cities, regions, countries
    location_patterns = [
        # US States
        r'\b(California|Texas|New York|Florida|Illinois|Pennsylvania|Ohio|Georgia|North Carolina|Michigan)\b',
        r'\b(Virginia|Washington|Arizona|Massachusetts|Tennessee|Indiana|Missouri|Maryland|Wisconsin|Colorado)\b',
        r'\b(Minnesota|Louisiana|Alabama|Kentucky|Oregon|Oklahoma|Connecticut|Iowa|Nevada|Arkansas)\b',
        r'\b(Alaska|Hawaii|Maine|Vermont|New Hampshire|Rhode Island|Delaware|Montana|Wyoming|Utah)\b',

        # Major cities
        r'\b(Houston|Dallas|Austin|San Antonio|Los Angeles|San Francisco|San Diego|Phoenix|Seattle|Portland)\b',
        r'\b(Denver|Las Vegas|Chicago|New York City|Boston|Philadelphia|Atlanta|Miami|Detroit|Cleveland)\b',

        # Regions
        r'\b(Pacific Northwest|Southwest|Midwest|Northeast|Southeast|West Coast|East Coast|Gulf Coast)\b',
        r'\b(Western states|Eastern states|Southern states|Northern states)\b',

        # Countries/Provinces
        r'\b(Canada|Mexico|Alberta|British Columbia|Ontario|Quebec)\b'
    ]

    for pattern in location_patterns:
        entities["locations"].extend(re.findall(pattern, text, re.IGNORECASE))

    # DATES - Multiple date formats
    date_patterns = [
        r'\b\d{1,2}[./\-]\d{1,2}[./\-]\d{4}\b',  # MM/DD/YYYY, MM-DD-YYYY
        r'\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b',  # YYYY/MM/DD
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:Q[1-4]|Quarter\s+[1-4])\s+\d{4}\b',  # Q1 2001, Quarter 1 2001
        r'\b\d{4}\b(?=\s*(?:year|fiscal|calendar))',  # Year references
    ]

    for pattern in date_patterns:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))

    # PROJECTS - ONLY actual project names
    project_patterns = [
        r'\b(Project\s+[A-Z][A-Za-z0-9\s]{2,20})\b',  # Project Boomerang, Project Alpha
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Project)\b',  # Whitewing Project, Atlantic Project
        r'\b(Project\s+[A-Z]+)\b',  # Project MEGS, Project LJM
        # Special named projects from Enron context
        r'\b(Whitewing|Marlin|Atlantic|Osprey|Braveheart|Yosemite|MEGS|Margaux|Backbone|Nahanni|Moose|Fishtail|Blackhawk)\b',
        r'\b(Chewco|JEDI\s+[IVX]+|Raptor)\b',  # JEDI I, JEDI II, Raptor structures
        r'\b(LJM\s+[IVX\d]*)\b',  # LJM entities
    ]

    for pattern in project_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            project = match if isinstance(match, str) else match[0] if match else ""
            if project and len(project.strip()) > 2:
                entities["projects"].append(project.strip())

    # LEGAL - Bills, acts, sections, orders, dockets, legal references
    legal_patterns = [
        r'\b([A-Z]{2}\s*\d{3,5})\b',  # Bill patterns: AB1890, SB123, HR456
        r'\b(Section\s+\d+(?:\([a-z]\))?)\b',  # Section references
        r'\b(Order\s+(?:No\.?\s*)?\d+)\b',  # Regulatory orders
        r'\b(FERC\s+Order\s+\d+)\b',
        r'\b(Docket\s+(?:No\.?\s*)?[A-Z]{1,3}\d{2,3}-\d{1,4}(?:-\d{1,4})?)\b',  # Docket numbers
        r'\b([A-Z]{3,6}\s+(?:Act|Bill|Code|Statute|Regulation))\b',  # Acts and bills
        r'\b(Federal\s+Power\s+Act|Public\s+Utility\s+Holding\s+Company\s+Act)\b',
        r'\b(Bankruptcy\s+(?:Code|Act|Order)|Sarbanes-Oxley)\b',
        r'\b(Rule\s+\d+[A-Za-z]*)\b',  # Rule 10b-5, Rule 506
        r'\b(USC\s+§\s*\d+)\b',  # US Code sections
    ]

    for pattern in legal_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            legal_item = match if isinstance(match, str) else match[0] if match else ""
            if legal_item and len(legal_item.strip()) > 1:
                entities["legal"].append(legal_item.strip())

    # TOPICS - Comprehensive business/energy/legal terms
    topic_categories = {
        'energy': ['electricity', 'power', 'energy', 'gas', 'oil', 'renewable', 'solar', 'wind', 'coal', 'nuclear'],
        'trading': ['trading', 'market', 'price', 'pricing', 'swap', 'hedge', 'derivative', 'commodity', 'futures'],
        'regulatory': ['regulation', 'regulatory', 'compliance', 'deregulation', 'oversight', 'jurisdiction', 'tariff'],
        'financial': ['revenue', 'profit', 'loss', 'investment', 'finance', 'accounting', 'budget', 'cost', 'expense'],
        'legal': ['lawsuit', 'litigation', 'contract', 'agreement', 'settlement', 'arbitration', 'dispute', 'claim'],
        'operations': ['transmission', 'distribution', 'generation', 'capacity', 'reliability', 'outage',
                       'maintenance'],
        'crisis': ['crisis', 'emergency', 'alert', 'critical', 'urgent', 'shortage', 'blackout', 'shortage']
    }

    text_lower = text.lower()
    for category, words in topic_categories.items():
        for word in words:
            if word in text_lower:
                entities["topics"].append(word)

    # Clean up and remove duplicates
    for key in entities:
        # Remove duplicates while preserving order, filter empty/short items
        seen = set()
        cleaned_items = []
        for item in entities[key]:
            item_clean = item.strip()
            if item_clean and len(item_clean) > 1 and item_clean.lower() not in seen:
                seen.add(item_clean.lower())
                cleaned_items.append(item_clean)
        entities[key] = cleaned_items[:15]  # Limit to 15 items per category

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
    results = []  # Changed from dict to list
    total_emails = len(emails)

    for batch_start in range(0, total_emails, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_emails)
        batch_emails = emails[batch_start:batch_end]

        print(f"\n--- Processing Batch {batch_start // BATCH_SIZE + 1} (emails {batch_start + 1}-{batch_end}) ---")

        for i, email_data in enumerate(batch_emails):
            email_num = batch_start + i + 1
            print(f"\nAnalyzing email {email_num}...")
            email_result = analyze_email(email_data)
            email_result["email_id"] = email_num  # Add email ID as a field
            results.append(email_result)  # Append to list instead of dict
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
    for email_result in results:
        email_id = email_result.get("email_id", "Unknown")
        classification = email_result.get("classification", "Unknown")
        tone = email_result.get("tone_analysis", "Unknown")
        print(f"  Email {email_id}: {classification} - {tone}")


if __name__ == "__main__":
    main()
