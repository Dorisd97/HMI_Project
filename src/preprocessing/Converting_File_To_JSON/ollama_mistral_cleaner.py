"""
Ollama Mistral Email Cleaner - JSON Input Only
Clean version that works with JSON email data
"""

import re
import os
import json
import subprocess
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional

# Try to import config
try:
    from src.config import config

    BASE_DIR = config.BASE_DIR
    # Use different paths for JSON input/output
    INPUT_JSON = config.CLEANED_JSON_PATH  # Source JSON file
    OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'enron_complete_cleaned2.json')  # Output JSON file
    print(f"✅ Using config.py - Base directory: {BASE_DIR}")
except ImportError:
    print("⚠️ Config not found, using current directory")
    BASE_DIR = os.getcwd()
    INPUT_JSON = 'cleaned_enron.json'
    OUTPUT_JSON = 'enron_complete_cleaned2.json'

# Ollama Configuration for Mistral
OLLAMA_CONFIG = {
    "model": "mistral",  # Can also use "mistral:7b", "mistral:instruct", etc.
    "base_url": "http://localhost:11434",
    "temperature": 0.1,
    "timeout": 60,  # Longer timeout for local processing
    "max_tokens": 2000,
    "batch_delay": 1,  # Delay between requests to prevent overload
    "retry_count": 3,
    "retry_delay": 5
}


class OllamaMistralCleaner:
    def __init__(self):
        """Initialize Ollama Mistral cleaner"""
        self.config = OLLAMA_CONFIG
        self.processed_count = 0
        self.failed_count = 0

        # Check if Ollama is running
        self.check_ollama_status()

        # Ensure Mistral model is available
        self.ensure_mistral_model()

    def check_ollama_status(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.config['base_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service is running")
                return True
            else:
                print("❌ Ollama service not responding properly")
                return False
        except requests.exceptions.RequestException:
            print("❌ Ollama service not running!")
            print("🔧 To start Ollama:")
            print("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   2. Start: ollama serve")
            print("   3. Run this script again")
            return False

    def ensure_mistral_model(self):
        """Ensure Mistral model is downloaded"""
        try:
            # Check available models
            response = requests.get(f"{self.config['base_url']}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]

                # Check if Mistral is available
                mistral_available = any('mistral' in name.lower() for name in model_names)

                if mistral_available:
                    print("✅ Mistral model is available")
                    # Find the exact model name
                    for name in model_names:
                        if 'mistral' in name.lower():
                            self.config['model'] = name.split(':')[0]  # Use base name
                            print(f"🤖 Using model: {name}")
                            break
                else:
                    print("⬇️ Mistral model not found. Downloading...")
                    self.download_mistral_model()
            else:
                print("❌ Cannot check available models")

        except Exception as e:
            print(f"⚠️ Error checking models: {str(e)}")

    def download_mistral_model(self):
        """Download Mistral model using Ollama"""
        print("📥 Downloading Mistral model (this may take a while)...")
        try:
            # Try to pull mistral model
            result = subprocess.run(['ollama', 'pull', 'mistral'],
                                    capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                print("✅ Mistral model downloaded successfully")
            else:
                print(f"❌ Failed to download Mistral: {result.stderr}")
                print("🔧 Try manually: ollama pull mistral")

        except subprocess.TimeoutExpired:
            print("⏰ Download taking longer than expected, continuing anyway...")
        except FileNotFoundError:
            print("❌ Ollama CLI not found. Please install Ollama first.")
        except Exception as e:
            print(f"⚠️ Download error: {str(e)}")

    def load_emails_from_json(self, json_file: str, limit: int = None) -> List[Dict]:
        """Load emails from JSON file"""
        try:
            if not os.path.exists(json_file):
                print(f"❌ JSON file not found: {json_file}")
                return []

            print(f"📂 Loading emails from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                emails = data
            elif isinstance(data, dict):
                # Check common keys for email arrays
                if 'emails' in data:
                    emails = data['emails']
                elif 'data' in data:
                    emails = data['data']
                elif 'items' in data:
                    emails = data['items']
                else:
                    # Try to find the largest array in the JSON
                    arrays = {k: v for k, v in data.items() if isinstance(v, list)}
                    if arrays:
                        largest_key = max(arrays.keys(), key=lambda k: len(arrays[k]))
                        emails = arrays[largest_key]
                        print(f"🔍 Using '{largest_key}' array with {len(emails)} items")
                    else:
                        print("❌ No email array found in JSON")
                        return []
            else:
                print("❌ Invalid JSON structure")
                return []

            total_emails = len(emails)
            print(f"📧 Found {total_emails} emails in JSON")

            # Limit if requested
            if limit and limit < total_emails:
                emails = emails[:limit]
                print(f"🧪 Limited to first {limit} emails for testing")

            return emails

        except Exception as e:
            print(f"❌ Error loading JSON: {str(e)}")
            return []

    def extract_email_content_from_json(self, email_data: Dict) -> Dict[str, str]:
        """Extract email content from JSON email object"""
        try:
            # Handle different JSON email formats
            raw_data = {}

            # Try different common field names for headers
            header_mappings = {
                'subject': ['subject', 'Subject', 'title', 'Title'],
                'from': ['from', 'From', 'sender', 'Sender', 'from_email'],
                'to': ['to', 'To', 'recipient', 'Recipient', 'to_email', 'recipients'],
                'date': ['date', 'Date', 'timestamp', 'Timestamp', 'sent_date', 'created_date']
            }

            for field, possible_keys in header_mappings.items():
                for key in possible_keys:
                    if key in email_data:
                        raw_data[field] = str(email_data[key])
                        break
                if field not in raw_data:
                    raw_data[field] = ""

            # Try different common field names for body
            body_keys = ['body', 'Body', 'content', 'Content', 'text', 'Text', 'message', 'Message']
            body = ""
            for key in body_keys:
                if key in email_data and email_data[key]:
                    body = str(email_data[key])
                    break

            raw_data['body'] = body

            # If no standard fields found, try to extract from nested structures
            if not any(raw_data.values()):
                # Try headers sub-object
                if 'headers' in email_data:
                    headers = email_data['headers']
                    raw_data.update({
                        'subject': str(headers.get('subject', headers.get('Subject', ''))),
                        'from': str(headers.get('from', headers.get('From', ''))),
                        'to': str(headers.get('to', headers.get('To', ''))),
                        'date': str(headers.get('date', headers.get('Date', '')))
                    })

                # Try content sub-object
                if 'content' in email_data:
                    content = email_data['content']
                    if isinstance(content, dict):
                        raw_data['body'] = str(content.get('body', content.get('text', content.get('message', ''))))
                    else:
                        raw_data['body'] = str(content)

            return raw_data

        except Exception as e:
            print(f"Error extracting JSON email: {str(e)}")
            return None

    def create_mistral_prompt(self, raw_email: Dict[str, str]) -> str:
        """Create optimized prompt for Mistral model"""
        # Limit body length for better processing
        body = raw_email.get('body', '')[:3000]

        prompt = f"""<s>[INST] You are an expert email data processor. Clean and analyze this email data.

EMAIL DATA:
Subject: {raw_email.get('subject', 'N/A')}
From: {raw_email.get('from', 'N/A')}
To: {raw_email.get('to', 'N/A')}
Date: {raw_email.get('date', 'N/A')}

Body:
{body}

TASK: Clean this email and extract structured information. Respond with ONLY a valid JSON object following this exact structure:

{{
  "cleaned_headers": {{
    "date": "readable date",
    "from": "clean sender", 
    "to": "clean recipients",
    "subject": "clean subject"
  }},
  "content": {{
    "main_message": "cleaned main content",
    "forwarded_content": "forwarded content or null",
    "summary": "brief summary",
    "word_count": number
  }},
  "entities": {{
    "people": ["person names"],
    "organizations": ["companies/orgs"], 
    "locations": ["places"],
    "amounts": ["money amounts"],
    "dates": ["specific dates"],
    "topics": ["main themes"]
  }},
  "analysis": {{
    "type": "business/personal/alert/legal/meeting/other",
    "tone": "formal/informal/urgent/neutral",
    "has_forwarded": true/false,
    "language": "english/other",
    "confidence": 0.8
  }}
}}

INSTRUCTIONS:
- Remove email headers and technical artifacts
- Clean formatting but keep meaning
- Extract real entities mentioned in content
- Separate forwarded content if present
- Classify email type and tone accurately
- Return ONLY valid JSON, no other text [/INST]"""

        return prompt

    def call_mistral(self, prompt: str) -> Optional[Dict]:
        """Call Ollama Mistral with retry logic"""
        for attempt in range(self.config["retry_count"]):
            try:
                data = {
                    "model": self.config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config["temperature"],
                        "num_predict": self.config["max_tokens"]
                    }
                }

                response = requests.post(
                    f"{self.config['base_url']}/api/generate",
                    json=data,
                    timeout=self.config["timeout"]
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("response", "")
                    return self.parse_response(content)
                else:
                    raise Exception(f"Ollama error: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"   Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config["retry_count"] - 1:
                    time.sleep(self.config["retry_delay"])
                else:
                    return None
        return None

    def parse_response(self, content: str) -> Optional[Dict]:
        """Parse Mistral response and extract JSON"""
        try:
            # Clean the response
            content = content.strip()

            # Find JSON block
            json_start = content.find('{')
            json_end = content.rfind('}')

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = content[json_start:json_end + 1]

                # Clean up common Mistral formatting issues
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control chars
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays

                return json.loads(json_str)
            else:
                print(f"   No valid JSON found in response")
                return None

        except json.JSONDecodeError as e:
            print(f"   JSON parsing error: {str(e)}")
            print(f"   Response preview: {content[:200]}...")
            return None
        except Exception as e:
            print(f"   Parse error: {str(e)}")
            return None

    def process_json_email_with_mistral(self, email_data: Dict, email_index: int) -> Optional[Dict]:
        """Process single email from JSON with Mistral"""
        # Extract email content from JSON
        raw_email = self.extract_email_content_from_json(email_data)
        if not raw_email:
            return None

        # Create optimized prompt
        prompt = self.create_mistral_prompt(raw_email)

        # Call Mistral
        result = self.call_mistral(prompt)
        if not result:
            return None

        # Add processing metadata
        result["processing_info"] = {
            "source_email_index": email_index,
            "original_data_keys": list(email_data.keys()),
            "processed_with": f"ollama/{self.config['model']}",
            "processed_at": datetime.now().isoformat()
        }

        # Include original data for reference if needed
        result["original_email"] = {
            "subject": raw_email.get('subject', ''),
            "from": raw_email.get('from', ''),
            "to": raw_email.get('to', ''),
            "date": raw_email.get('date', '')
        }

        return result

    def process_emails_from_json(self, input_json_file: str = None, test_mode: bool = True, test_limit: int = 5):
        """Main processing function for JSON input"""
        print("🤖 Ollama Mistral Email Cleaner (JSON Input)")
        print("=" * 60)
        print(f"🧠 Model: {self.config['model']}")
        print(f"🌐 Endpoint: {self.config['base_url']}")

        # Use provided file or default
        json_file = input_json_file or INPUT_JSON

        # Create output directory
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

        # Load emails from JSON
        limit = test_limit if test_mode else None
        emails = self.load_emails_from_json(json_file, limit)

        if not emails:
            print("❌ No emails to process")
            return None

        if test_mode:
            print(f"🧪 TEST MODE: Processing {len(emails)} emails")
        else:
            print(f"📧 Processing {len(emails)} emails from JSON")
            confirm = input("⚠️ This will take a long time. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Cancelled")
                return None

        # Process emails
        processed_emails = []

        print("🔄 Processing emails with Mistral...")
        start_time = time.time()

        for i, email_data in enumerate(emails, 1):
            # Get a preview of the email for logging
            subject_preview = ""
            if isinstance(email_data, dict):
                # Try to find subject for preview
                for key in ['subject', 'Subject', 'title']:
                    if key in email_data:
                        subject_preview = str(email_data[key])[:30]
                        break

            print(f"   📧 [{i}/{len(emails)}] {subject_preview}...")

            try:
                cleaned_email = self.process_json_email_with_mistral(email_data, i - 1)
                if cleaned_email:
                    processed_emails.append(cleaned_email)
                    self.processed_count += 1
                    print(f"      ✅ Success")
                else:
                    self.failed_count += 1
                    print(f"      ❌ Failed")

                # Small delay to prevent overload
                time.sleep(self.config["batch_delay"])

            except Exception as e:
                self.failed_count += 1
                print(f"      ❌ Error: {str(e)}")

        processing_time = time.time() - start_time

        # Create final result
        result = {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "model_used": f"ollama/{self.config['model']}",
                "processing_time_seconds": round(processing_time, 1),
                "source_file": json_file,
                "total_emails_in_source": len(emails),
                "successfully_processed": self.processed_count,
                "failed_emails": self.failed_count,
                "test_mode": test_mode,
                "test_limit": test_limit if test_mode else None
            },
            "summary": {
                "total_emails": len(processed_emails),
                "success_rate": round((self.processed_count / len(emails)) * 100, 1) if emails else 0,
                "avg_processing_time": round(processing_time / len(emails), 1) if emails else 0,
                "email_types": self.get_distribution(processed_emails, 'analysis', 'type'),
                "tones": self.get_distribution(processed_emails, 'analysis', 'tone'),
                "languages": self.get_distribution(processed_emails, 'analysis', 'language')
            },
            "emails": processed_emails
        }

        # Save result
        if test_mode:
            output_file = OUTPUT_JSON.replace('.json', '_mistral_test.json')
        else:
            output_file = OUTPUT_JSON.replace('.json', '_mistral.json')

        print(f"\n💾 Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Show results
        print("\n🎉 JSON PROCESSING COMPLETE!")
        print(f"📊 Results:")
        print(f"   📁 Source: {json_file}")
        print(f"   ✅ Processed: {self.processed_count} emails")
        print(f"   ❌ Failed: {self.failed_count} emails")
        print(f"   📈 Success rate: {result['summary']['success_rate']}%")
        print(f"   ⏱️ Total time: {processing_time / 60:.1f} minutes")
        print(f"   ⚡ Avg per email: {result['summary']['avg_processing_time']:.1f} seconds")
        print(f"   📁 Saved to: {output_file}")

        # Show sample
        if processed_emails:
            sample = processed_emails[0]
            print(f"\n📋 Sample result:")
            print(f"   Original from: {sample.get('original_email', {}).get('from', 'N/A')}")
            print(f"   Cleaned from: {sample.get('cleaned_headers', {}).get('from', 'N/A')}")
            print(f"   Original subject: {sample.get('original_email', {}).get('subject', 'N/A')[:50]}...")
            print(f"   Cleaned subject: {sample.get('cleaned_headers', {}).get('subject', 'N/A')[:50]}...")
            print(f"   Type: {sample.get('analysis', {}).get('type', 'N/A')}")
            print(
                f"   Entities: {len(sample.get('entities', {}).get('people', []))} people, {len(sample.get('entities', {}).get('organizations', []))} orgs")

        return result

    def get_distribution(self, emails, section, field):
        """Get distribution of a field"""
        dist = {}
        for email in emails:
            value = email.get(section, {}).get(field, 'unknown')
            dist[value] = dist.get(value, 0) + 1
        return dist


def create_sample_json():
    """Create a sample JSON file for testing"""
    sample_emails = [
        {
            "subject": "Project Alpha - Q4 Results",
            "from": "jeff.skilling@enron.com",
            "to": "kenneth.lay@enron.com",
            "date": "2001-01-15 09:30:00",
            "body": """Ken,

Our trading operations generated $2.3 billion in revenue this quarter. The California market continues to be volatile with FERC investigating our practices.

Key highlights:
- West Coast trading: $1.2B 
- Gas operations: $800M
- Risk management improvements implemented

We should discuss the regulatory situation at tomorrow's board meeting.

Best,
Jeff Skilling
President & COO"""
        },
        {
            "subject": "URGENT: California Investigation Update",
            "from": "susan.mara@enron.com",
            "to": "jeff.dasovich@enron.com,james.steffes@enron.com",
            "date": "2001-04-25 14:22:00",
            "body": """Team,

FERC is expanding their investigation into our California trading practices. Legal recommends immediate document preservation.

----- Forwarded from regulatory@ferc.gov -----
Subject: Formal Investigation Notice

Enron Corp is hereby notified of formal investigation into trading practices during California energy crisis. All relevant documents must be preserved.

Investigation scope:
- Power trading activities 2000-2001
- Price manipulation allegations  
- Market manipulation claims
- Coordination with other generators

Please coordinate response with legal team immediately.

Susan Mara
Government Affairs"""
        },
        {
            "subject": "Re: Dabhol Project Status",
            "from": "rebecca.mark@enron.com",
            "to": "jeff.skilling@enron.com",
            "date": "2000-11-08 11:45:00",
            "body": """Jeff,

The Dabhol situation in India is deteriorating rapidly. The Maharashtra state government is refusing to honor power purchase agreements.

Current status:
- $2.9 billion invested to date
- Phase I operational but payments delayed
- Phase II construction halted
- Local political opposition increasing

We may need to consider write-downs or exit strategies. The Indian government is not providing the support promised.

Rebecca"""
        },
        {
            "subject": "Analyst Call Preparation",
            "from": "andrew.fastow@enron.com",
            "to": "jeff.skilling@enron.com,kenneth.lay@enron.com",
            "date": "2001-07-12 16:30:00",
            "body": """Ken and Jeff,

Preparing for next week's analyst call. Need to discuss how we present the following:

Financial Highlights:
- Revenue: $50.1 billion (up 151%)
- Net income: $979 million 
- Funds flow from operations: $4.8 billion
- Total assets: $65.5 billion

Key talking points:
- Strong performance across all business units
- Continued growth in wholesale energy operations
- Broadband services expansion
- International investments

The Street is expecting strong guidance for Q3. We should emphasize our asset-light business model and recurring cash flows.

Andy Fastow
CFO"""
        },
        {
            "subject": "Enron Online Platform Metrics",
            "from": "louise.kitchen@enron.com",
            "to": "jeff.skilling@enron.com",
            "date": "2000-12-15 10:15:00",
            "body": """Jeff,

Enron Online continues to exceed expectations. December metrics:

Platform Performance:
- Daily transactions: $2.5 billion average
- Total volume since launch: $350 billion
- Active counterparties: 1,400+
- Products traded: 1,800+

Geographic breakdown:
- North America: 78%
- Europe: 15% 
- Asia: 4%
- Other: 3%

The platform is capturing significant market share in natural gas and power trading. Margins remain strong due to our information advantage.

Recommend continued investment in technology infrastructure and expansion into new commodities.

Louise Kitchen
President, Enron Online"""
        }
    ]

    json_data = {
        "metadata": {
            "source": "Sample Enron emails for testing",
            "created": datetime.now().isoformat(),
            "total_emails": len(sample_emails)
        },
        "emails": sample_emails
    }

    # Save to input file
    with open(INPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Created sample JSON with {len(sample_emails)} emails: {INPUT_JSON}")
    return INPUT_JSON


def setup_ollama():
    """Help user setup Ollama and Mistral"""
    print("🔧 Ollama Setup Guide")
    print("=" * 30)

    print("\n1️⃣ Install Ollama:")
    print("   Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   Windows: Download from https://ollama.ai")

    print("\n2️⃣ Start Ollama service:")
    print("   ollama serve")

    print("\n3️⃣ Install Mistral model:")
    print("   ollama pull mistral")
    print("   (This downloads ~4GB, may take time)")

    print("\n4️⃣ Test installation:")
    print("   ollama run mistral")
    print("   Type: Hello! (should get response)")
    print("   Type: /bye (to exit)")

    print("\n5️⃣ Run this script:")
    print("   python clean_ollama_mistral_cleaner.py")


def main():
    """Main execution function"""
    print("🤖 Ollama Mistral Email Cleaner (JSON Input)")
    print("=" * 60)

    # Check if user needs setup help
    setup_help = input("📚 Need Ollama setup help? (y/n, default=n): ").lower() == 'y'
    if setup_help:
        setup_ollama()
        return

    # Initialize cleaner
    cleaner = OllamaMistralCleaner()

    # Check for input JSON file
    input_file = input(f"📁 Input JSON file path (default: {INPUT_JSON}): ").strip()
    if not input_file:
        input_file = INPUT_JSON

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        create_sample = input("📄 Create sample JSON file for testing? (y/n, default=y): ").lower() != 'n'
        if create_sample:
            input_file = create_sample_json()
        else:
            print("❌ Cannot proceed without input file")
            return

    # Get test parameters
    test_mode = input("\n🧪 Run in test mode? (y/n, default=y): ").lower() != 'n'

    if test_mode:
        test_limit = int(input("📧 How many emails to test? (default=5): ") or "5")
        print(f"\n🚀 Starting test with {test_limit} emails from JSON...")
        result = cleaner.process_emails_from_json(input_file, test_mode=True, test_limit=test_limit)
    else:
        print("\n⚠️ Full processing will take a long time with local model")
        print("💡 Recommendation: Start with test mode first")
        confirm = input("Continue with full processing? (y/n): ")
        if confirm.lower() == 'y':
            result = cleaner.process_emails_from_json(input_file, test_mode=False)
        else:
            print("❌ Cancelled")
            return

    if result:
        print(f"\n✨ Success! Mistral has intelligently cleaned your emails from JSON.")
        print(f"🔍 JSON Processing Benefits:")
        print(f"   ✅ Structured input - handles any JSON email format")
        print(f"   ✅ Easy testing - just specify number of emails")
        print(f"   ✅ No file extraction needed")
        print(f"   ✅ Preserves original vs cleaned comparison")
        print(f"   ✅ Fast and local processing")

        # Show comparison
        if result['emails']:
            sample = result['emails'][0]
            original = sample.get('original_email', {})
            cleaned = sample.get('cleaned_headers', {})

            print(f"\n📊 Example Before/After:")
            print(f"   Original subject: {original.get('subject', 'N/A')[:50]}")
            print(f"   Cleaned subject:  {cleaned.get('subject', 'N/A')[:50]}")
            print(
                f"   Analysis: {sample.get('analysis', {}).get('type', 'N/A')} email, {sample.get('analysis', {}).get('tone', 'N/A')} tone")


if __name__ == "__main__":
    main()
