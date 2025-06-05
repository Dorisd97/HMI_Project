#!/usr/bin/env python3
"""
Ollama Mistral Email Cleaner
Uses local Ollama with Mistral model to clean and analyze emails
"""

import re
import os
import json
import zipfile
import requests
import time
import subprocess
from datetime import datetime
from email import message_from_string
from email.header import decode_header
from typing import Dict, List, Optional

# Try to import config
try:
    from src.config import config

    BASE_DIR = config.BASE_DIR
    ZIP_PATH = config.ZIP_PATH
    UNZIP_DIR = config.UNZIP_DIR_TEST
    OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'enron_complete_test.json')
    print(f"✅ Using config.py - Base directory: {BASE_DIR}")
except ImportError:
    print("⚠️ Config not found, using current directory")
    BASE_DIR = os.getcwd()
    ZIP_PATH = os.path.join(BASE_DIR, 'Enron.zip')
    UNZIP_DIR = os.path.join(BASE_DIR, 'test_data')
    OUTPUT_JSON = 'enron_complete_test.json'

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

    def extract_zip(self):
        """Extract Enron.zip if it exists"""
        if os.path.exists(ZIP_PATH):
            print(f"📦 Extracting {ZIP_PATH}...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(UNZIP_DIR)
            print(f"✅ Extracted to {UNZIP_DIR}")
            return True
        else:
            print(f"❌ ZIP file not found: {ZIP_PATH}")
            return False

    def find_email_files(self, directory: str, limit: int = None) -> List[str]:
        """Find email files with optional limit"""
        email_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.txt', '.eml')):
                    email_files.append(os.path.join(root, file))
                    if limit and len(email_files) >= limit:
                        return email_files
        return email_files

    def extract_raw_email_content(self, file_path: str) -> Dict[str, str]:
        """Extract raw email headers and body"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()

            # Parse email
            email_msg = message_from_string(email_content)

            # Extract headers
            raw_data = {
                "subject": self.decode_header_value(email_msg.get('Subject', '')),
                "from": self.decode_header_value(email_msg.get('From', '')),
                "to": self.decode_header_value(email_msg.get('To', '')),
                "date": self.decode_header_value(email_msg.get('Date', '')),
                "body": self.extract_body(email_msg)
            }

            return raw_data

        except Exception as e:
            print(f"Error extracting {file_path}: {str(e)}")
            return None

    def decode_header_value(self, header_value):
        """Decode email header"""
        if not header_value:
            return ""
        try:
            decoded_pairs = decode_header(header_value)
            decoded_string = ""
            for value, encoding in decoded_pairs:
                if isinstance(value, bytes):
                    decoded_string += value.decode(encoding or 'utf-8', errors='ignore')
                else:
                    decoded_string += value
            return decoded_string.strip()
        except:
            return str(header_value).strip()

    def extract_body(self, email_msg):
        """Extract email body"""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            return payload.decode('utf-8', errors='ignore')
                        except:
                            return str(payload)
        else:
            payload = email_msg.get_payload(decode=True)
            if payload:
                try:
                    return payload.decode('utf-8', errors='ignore')
                except:
                    return str(payload)
            else:
                return email_msg.get_payload()
        return ""

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

    def process_email_with_mistral(self, file_path: str) -> Optional[Dict]:
        """Process single email with Mistral"""
        # Extract raw email
        raw_email = self.extract_raw_email_content(file_path)
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
            "source_file": os.path.basename(file_path),
            "processed_with": f"ollama/{self.config['model']}",
            "processed_at": datetime.now().isoformat()
        }

        return result

    def process_all_emails(self, test_mode: bool = True, test_limit: int = 5):
        """Main processing function"""
        print("🤖 Ollama Mistral Email Cleaner")
        print("=" * 50)
        print(f"🧠 Model: {self.config['model']}")
        print(f"🌐 Endpoint: {self.config['base_url']}")

        # Create data directory
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

        # Extract ZIP if needed
        if not os.path.exists(UNZIP_DIR):
            if not self.extract_zip():
                print("❌ Cannot proceed without email data")
                return

        # Find email files
        limit = test_limit if test_mode else None
        print(f"🔍 Scanning for email files...")
        email_files = self.find_email_files(UNZIP_DIR, limit)

        if test_mode:
            print(f"🧪 TEST MODE: Processing {len(email_files)} emails")
        else:
            print(f"📧 Found {len(email_files)} email files")
            confirm = input("⚠️ This will take a long time. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Cancelled")
                return

        if not email_files:
            print("❌ No email files found")
            return

        # Process emails
        processed_emails = []

        print("🔄 Processing emails with Mistral...")
        start_time = time.time()

        for i, file_path in enumerate(email_files, 1):
            filename = os.path.basename(file_path)
            print(f"   📧 [{i}/{len(email_files)}] {filename[:30]}...")

            try:
                cleaned_email = self.process_email_with_mistral(file_path)
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
                "source_directory": UNZIP_DIR,
                "total_files_found": len(email_files),
                "successfully_processed": self.processed_count,
                "failed_files": self.failed_count,
                "test_mode": test_mode
            },
            "summary": {
                "total_emails": len(processed_emails),
                "success_rate": round((self.processed_count / len(email_files)) * 100, 1) if email_files else 0,
                "avg_processing_time": round(processing_time / len(email_files), 1) if email_files else 0,
                "email_types": self.get_distribution(processed_emails, 'analysis', 'type'),
                "tones": self.get_distribution(processed_emails, 'analysis', 'tone'),
                "languages": self.get_distribution(processed_emails, 'analysis', 'language')
            },
            "emails": processed_emails
        }

        # Save result
        output_file = OUTPUT_JSON.replace('.json', '_mistral_test.json') if test_mode else OUTPUT_JSON.replace('.json',
                                                                                                               '_mistral.json')
        print(f"\n💾 Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Show results
        print("\n🎉 PROCESSING COMPLETE!")
        print(f"📊 Results:")
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
            print(f"   From: {sample.get('cleaned_headers', {}).get('from', 'N/A')}")
            print(f"   Subject: {sample.get('cleaned_headers', {}).get('subject', 'N/A')[:50]}...")
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
    print("   python ollama_mistral_cleaner.py")


def main():
    """Main execution function"""
    print("🤖 Ollama Mistral Email Cleaner")
    print("=" * 50)

    # Check if user needs setup help
    setup_help = input("📚 Need Ollama setup help? (y/n, default=n): ").lower() == 'y'
    if setup_help:
        setup_ollama()
        return

    # Initialize cleaner
    cleaner = OllamaMistralCleaner()

    # Ask for test mode
    test_mode = input("\n🧪 Run in test mode? (y/n, default=y): ").lower() != 'n'

    if test_mode:
        test_limit = int(input("📧 How many emails to test? (default=5): ") or "5")
        print(f"\n🚀 Starting test with {test_limit} emails...")
        result = cleaner.process_all_emails(test_mode=True, test_limit=test_limit)
    else:
        print("\n⚠️ Full processing will take a long time with local model")
        print("💡 Recommendation: Start with test mode first")
        confirm = input("Continue with full processing? (y/n): ")
        if confirm.lower() == 'y':
            result = cleaner.process_all_emails(test_mode=False)
        else:
            print("❌ Cancelled")
            return

    if result:
        print(f"\n✨ Success! Mistral has intelligently cleaned your emails.")
        print(f"🔍 Benefits of local processing:")
        print(f"   ✅ No API costs")
        print(f"   ✅ Complete privacy (data never leaves your machine)")
        print(f"   ✅ No rate limits")
        print(f"   ✅ Works offline")
        print(f"   ⚠️ Slower than cloud APIs")


if __name__ == "__main__":
    main()
