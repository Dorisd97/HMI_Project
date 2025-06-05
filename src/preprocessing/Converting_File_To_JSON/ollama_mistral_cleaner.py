"""
Ollama Mistral Email Cleaner - JSON Input with BodyChain Support
Enhanced version that processes BodyChain email conversation threads
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
    INPUT_JSON = os.path.join(BASE_DIR, 'data', 'cleaned_body_chain_enron.json')  # Source JSON file
    OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'enron_complete_cleaned4.json')  # Output JSON file
    print(f"✅ Using config.py - Base directory: {BASE_DIR}")
except ImportError:
    print("⚠️ Config not found, using current directory")
    BASE_DIR = os.getcwd()
    INPUT_JSON = 'cleaned_body_chain_enron.json'
    OUTPUT_JSON = 'enron_complete_cleaned3.json'

# Ollama Configuration for Mistral
OLLAMA_CONFIG = {
    "model": "mistral",  # Can also use "mistral:7b", "mistral:instruct", etc.
    "base_url": "http://localhost:11434",
    "temperature": 0.1,
    "timeout": 120,  # Longer timeout for processing BodyChain
    "max_tokens": 3000,  # Increased for BodyChain processing
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
        """Extract email content from JSON email object including BodyChain"""
        try:
            # Handle different JSON email formats
            raw_data = {}

            # Standard header mappings for main email
            header_mappings = {
                'subject': ['Subject', 'subject', 'title', 'Title'],
                'from': ['From', 'from', 'sender', 'Sender', 'from_email'],
                'to': ['To', 'to', 'recipient', 'Recipient', 'to_email', 'recipients'],
                'date': ['Date', 'date', 'timestamp', 'Timestamp', 'sent_date', 'created_date'],
                'message_id': ['Message-ID', 'message_id', 'id'],
                'mime_version': ['Mime-Version', 'mime_version'],
                'content_type': ['Content-Type', 'content_type']
            }

            for field, possible_keys in header_mappings.items():
                for key in possible_keys:
                    if key in email_data:
                        raw_data[field] = str(email_data[key])
                        break
                if field not in raw_data:
                    raw_data[field] = ""

            # Extract main email body
            body_keys = ['Body', 'body', 'content', 'Content', 'text', 'Text', 'message', 'Message']
            body = ""
            for key in body_keys:
                if key in email_data and email_data[key]:
                    body = str(email_data[key])
                    break

            raw_data['body'] = body

            # Extract BodyChain if present
            body_chain = []
            if 'BodyChain' in email_data and isinstance(email_data['BodyChain'], list):
                for chain_item in email_data['BodyChain']:
                    if isinstance(chain_item, dict):
                        chain_entry = {
                            'from': str(chain_item.get('From', '')),
                            'sent': str(chain_item.get('Sent', '')),
                            'to': str(chain_item.get('To', '')),
                            'cc': str(chain_item.get('Cc', '')),
                            'bcc': str(chain_item.get('Bcc', '')),
                            'subject': str(chain_item.get('Subject', '')),
                            'body': str(chain_item.get('Body', ''))
                        }
                        body_chain.append(chain_entry)

            raw_data['body_chain'] = body_chain
            raw_data['body_chain_count'] = len(body_chain)

            # Extract additional metadata
            raw_data['source_file'] = str(email_data.get('SourceFile', ''))
            raw_data['x_folder'] = str(email_data.get('X-Folder', ''))
            raw_data['x_origin'] = str(email_data.get('X-Origin', ''))

            return raw_data

        except Exception as e:
            print(f"Error extracting JSON email: {str(e)}")
            return None

    def create_mistral_prompt(self, raw_email: Dict[str, str]) -> str:
        """Create optimized prompt for Mistral model including BodyChain"""
        # Limit main body length for better processing
        main_body = raw_email.get('body', '')[:2000]

        # Prepare BodyChain text
        body_chain_text = ""
        body_chain = raw_email.get('body_chain', [])

        if body_chain:
            body_chain_text = "\n\nEMAIL CONVERSATION THREAD (BodyChain):\n"
            for i, chain_item in enumerate(body_chain, 1):
                body_chain_text += f"\n--- Email {i} in Thread ---\n"
                body_chain_text += f"From: {chain_item.get('from', 'N/A')}\n"
                body_chain_text += f"To: {chain_item.get('to', 'N/A')}\n"
                body_chain_text += f"Subject: {chain_item.get('subject', 'N/A')}\n"
                body_chain_text += f"Body: {chain_item.get('body', '')[:1000]}\n"

        prompt = f"""<s>[INST] You are an expert email data processor. Clean and analyze this email data including conversation threads.

MAIN EMAIL:
Message-ID: {raw_email.get('message_id', 'N/A')}
Subject: {raw_email.get('subject', 'N/A')}
From: {raw_email.get('from', 'N/A')}
To: {raw_email.get('to', 'N/A')}
Date: {raw_email.get('date', 'N/A')}
Source File: {raw_email.get('source_file', 'N/A')}

Main Body:
{main_body}

{body_chain_text}

TASK: Clean this email and analyze the complete conversation thread. Respond with ONLY a valid JSON object following this exact structure:

{{
  "cleaned_headers": {{
    "date": "readable date",
    "from": "clean sender", 
    "to": "clean recipients",
    "subject": "clean subject",
    "message_id": "message ID"
  }},
  "content": {{
    "main_message": "cleaned main content",
    "forwarded_content": "forwarded content or null",
    "summary": "brief summary of entire conversation",
    "word_count": number,
    "thread_summary": "summary of conversation thread if BodyChain exists"
  }},
  "body_chain_analysis": {{
    "has_conversation_thread": true/false,
    "thread_length": number,
    "participants": ["list of email participants"],
    "thread_topics": ["main topics discussed in thread"],
    "conversation_flow": "description of how conversation progressed"
  }},
  "entities": {{
    "people": ["person names from all emails"],
    "organizations": ["companies/orgs from all emails"], 
    "locations": ["places mentioned"],
    "amounts": ["money amounts"],
    "dates": ["specific dates mentioned"],
    "topics": ["main themes across conversation"],
    "projects": ["project names mentioned"]
  }},
  "analysis": {{
    "type": "business/personal/alert/legal/meeting/project/other",
    "tone": "formal/informal/urgent/neutral",
    "has_forwarded": true/false,
    "is_conversation_thread": true/false,
    "language": "english/other",
    "confidence": 0.8,
    "business_importance": "high/medium/low",
    "contains_sensitive_info": true/false
  }}
}}

INSTRUCTIONS:
- Analyze both main email AND conversation thread (BodyChain)
- Extract entities from entire conversation, not just main email
- Identify conversation patterns and participant relationships
- Separate main email content from thread discussion
- Clean formatting but preserve all meaningful information
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
        """Process single email from JSON with Mistral including BodyChain"""
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
            "processed_at": datetime.now().isoformat(),
            "has_body_chain": len(raw_email.get('body_chain', [])) > 0,
            "body_chain_length": len(raw_email.get('body_chain', []))
        }

        # Include original data for reference
        result["original_email"] = {
            "message_id": raw_email.get('message_id', ''),
            "subject": raw_email.get('subject', ''),
            "from": raw_email.get('from', ''),
            "to": raw_email.get('to', ''),
            "date": raw_email.get('date', ''),
            "source_file": raw_email.get('source_file', ''),
            "body_chain_count": raw_email.get('body_chain_count', 0)
        }

        return result

    def process_emails_from_json(self, input_json_file: str = None, test_mode: bool = True, test_limit: int = 5):
        """Main processing function for JSON input with BodyChain support"""
        print("🤖 Ollama Mistral Email Cleaner (JSON Input with BodyChain)")
        print("=" * 70)
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

        print("🔄 Processing emails with Mistral (including BodyChain analysis)...")
        start_time = time.time()

        for i, email_data in enumerate(emails, 1):
            # Get a preview of the email for logging
            subject_preview = str(email_data.get('Subject', email_data.get('subject', 'No Subject')))[:30]
            body_chain_count = len(email_data.get('BodyChain', []))

            chain_info = f" ({body_chain_count} thread emails)" if body_chain_count > 0 else ""
            print(f"   📧 [{i}/{len(emails)}] {subject_preview}{chain_info}...")

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

        # Calculate BodyChain statistics
        body_chain_stats = self.calculate_body_chain_stats(processed_emails)

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
                "test_limit": test_limit if test_mode else None,
                "body_chain_processing": True
            },
            "summary": {
                "total_emails": len(processed_emails),
                "success_rate": round((self.processed_count / len(emails)) * 100, 1) if emails else 0,
                "avg_processing_time": round(processing_time / len(emails), 1) if emails else 0,
                "email_types": self.get_distribution(processed_emails, 'analysis', 'type'),
                "tones": self.get_distribution(processed_emails, 'analysis', 'tone'),
                "languages": self.get_distribution(processed_emails, 'analysis', 'language'),
                "business_importance": self.get_distribution(processed_emails, 'analysis', 'business_importance'),
                "body_chain_statistics": body_chain_stats
            },
            "emails": processed_emails
        }

        # Save result
        if test_mode:
            output_file = OUTPUT_JSON.replace('.json', '_mistral_bodychain_test.json')
        else:
            output_file = OUTPUT_JSON.replace('.json', '_mistral_bodychain.json')

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

        # Show BodyChain statistics
        print(f"\n🔗 BodyChain Analysis:")
        print(f"   📧 Emails with conversation threads: {body_chain_stats['emails_with_threads']}")
        print(f"   📊 Average thread length: {body_chain_stats['avg_thread_length']:.1f}")
        print(f"   📈 Max thread length: {body_chain_stats['max_thread_length']}")
        print(f"   🔗 Total conversation emails: {body_chain_stats['total_thread_emails']}")

        # Show sample
        if processed_emails:
            sample = processed_emails[0]
            print(f"\n📋 Sample result:")
            print(f"   Original from: {sample.get('original_email', {}).get('from', 'N/A')}")
            print(f"   Cleaned from: {sample.get('cleaned_headers', {}).get('from', 'N/A')}")
            print(f"   Original subject: {sample.get('original_email', {}).get('subject', 'N/A')[:50]}...")
            print(f"   Cleaned subject: {sample.get('cleaned_headers', {}).get('subject', 'N/A')[:50]}...")
            print(f"   Type: {sample.get('analysis', {}).get('type', 'N/A')}")
            print(f"   Has thread: {sample.get('body_chain_analysis', {}).get('has_conversation_thread', False)}")
            print(f"   Thread length: {sample.get('body_chain_analysis', {}).get('thread_length', 0)}")

        return result

    def calculate_body_chain_stats(self, processed_emails: List[Dict]) -> Dict:
        """Calculate statistics about BodyChain processing"""
        stats = {
            "emails_with_threads": 0,
            "total_thread_emails": 0,
            "avg_thread_length": 0,
            "max_thread_length": 0,
            "thread_length_distribution": {},
            "most_common_participants": {},
            "conversation_types": {}
        }

        thread_lengths = []
        all_participants = []

        for email in processed_emails:
            body_chain_analysis = email.get('body_chain_analysis', {})

            if body_chain_analysis.get('has_conversation_thread', False):
                stats["emails_with_threads"] += 1

                thread_length = body_chain_analysis.get('thread_length', 0)
                thread_lengths.append(thread_length)
                stats["total_thread_emails"] += thread_length

                # Track participants
                participants = body_chain_analysis.get('participants', [])
                all_participants.extend(participants)

        if thread_lengths:
            stats["avg_thread_length"] = sum(thread_lengths) / len(thread_lengths)
            stats["max_thread_length"] = max(thread_lengths)

            # Thread length distribution
            for length in thread_lengths:
                length_range = f"{length} emails"
                stats["thread_length_distribution"][length_range] = stats["thread_length_distribution"].get(length_range, 0) + 1

        # Most common participants
        from collections import Counter
        participant_counts = Counter(all_participants)
        stats["most_common_participants"] = dict(participant_counts.most_common(10))

        return stats

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
    print("   python ollama_mistral_cleaner_updated.py")


def main():
    """Main execution function"""
    print("🤖 Ollama Mistral Email Cleaner (JSON Input with BodyChain)")
    print("=" * 70)

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
        print(f"\n✨ Success! Mistral has intelligently cleaned your emails with BodyChain analysis.")
        print(f"🔍 Enhanced JSON Processing Benefits:")
        print(f"   ✅ BodyChain conversation thread analysis")
        print(f"   ✅ Multi-email thread entity extraction")
        print(f"   ✅ Conversation flow analysis")
        print(f"   ✅ Participant relationship mapping")
        print(f"   ✅ Enhanced business context understanding")

        # Show BodyChain insights
        body_chain_stats = result['summary']['body_chain_statistics']
        print(f"\n🔗 BodyChain Insights:")
        print(f"   📧 Emails with threads: {body_chain_stats['emails_with_threads']}")
        print(f"   📊 Average thread length: {body_chain_stats['avg_thread_length']:.1f}")
        print(f"   🔗 Total conversation emails: {body_chain_stats['total_thread_emails']}")


if __name__ == "__main__":
    main()
