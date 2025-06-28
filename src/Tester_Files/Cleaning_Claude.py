import json
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

"""
Email Data Cleaner - JSON Output Focus

This script cleans noisy email data and outputs a standardized JSON structure:

Final JSON Output Structure:
{
  "metadata": {
    "total_emails": 45,
    "cleaning_timestamp": "2024-01-15T10:30:00",
    "data_format": "cleaned_email_dataset",
    "schema_version": "1.0",
    "fields": {
      "message_id": "Unique identifier for the email",
      "date": "Standardized timestamp (YYYY-MM-DD HH:MM:SS)", 
      "from": "List of sender email addresses",
      "to": "List of recipient email addresses",
      "subject": "Cleaned subject line",
      "body": "Cleaned email body content",
      "source_file": "Original source file reference",
      "original_length": "Character count before cleaning",
      "cleaned_length": "Character count after cleaning"
    }
  },
  "emails": [
    {
      "message_id": "22289471.1075843076105.JavaMail.evans@thyme",
      "date": "2000-04-14 09:58:00",
      "from": ["frank.vickers@enron.com"],
      "to": ["jeff.dasovich@enron.com"],
      "subject": "Project Boomerang", 
      "body": "Jeff, thanks for the response. Consider yourself a member of Project Boomerang...",
      "source_file": "1. 2.txt",
      "original_length": 2847,
      "cleaned_length": 1205
    }
  ]
}
"""


class EmailDataCleaner:
    def __init__(self):
        # Patterns for removing email noise
        self.forwarding_patterns = [
            r'Forwarded by.*?on \d{2}/\d{2}/\d{4}.*?\n',
            r'Original Message.*?\n',
            r'From:.*?Sent:.*?\n',
            r'-----Original Message-----.*?\n',
            r'From:\s+.*?\nSent:\s+.*?\n',
            r'_+\nFrom:.*?\n',
        ]

        self.signature_patterns = [
            r'\n--+\n.*',  # Signature separator
            r'\n_{5,}.*',  # Underscore separators
            r'This email and any files transmitted.*?are confidential.*',
            r'CONFIDENTIAL.*?PRIVILEGE.*',
            r'The information contained.*?confidential.*',
            r'\nRegards,?\n.*',
            r'\nBest,?\n.*',
            r'\nThanks,?\n.*',
            r'\nSincerely,?\n.*',
        ]

        self.technical_patterns = [
            r'<.*?>',  # HTML/XML tags
            r'Message-ID:.*?\n',
            r'X-.*?:.*?\n',
            r'Mime-Version:.*?\n',
            r'Content-Type:.*?\n',
            r'Content-Transfer-Encoding:.*?\n',
            r'=\d{2}[A-F0-9]',  # Encoding artifacts like =01, =20
            r'<IMCEANOTES-.*?>',  # Exchange server artifacts
            r'mailto:.*?\s',
        ]

        self.attachment_patterns = [
            r'See attached file:.*?\)',
            r'Attached is.*?\n',
            r'Please see attached.*?\n',
            r'<<.*?>>',  # Attachment references
            r'\(See attached file:.*?\)',
            r'Attachment.*?:.*?\n',
        ]

        self.phone_email_patterns = [
            r'\(\d{3}\)\s?\d{3}-\d{4}',  # Phone numbers
            r'\d{3}-\d{3}-\d{4}',
            r'\d{3}\.\d{3}\.\d{4}',
            r'\d{3}\s\d{3}\s\d{4}',
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses in body
        ]

    def clean_email_body(self, body: str) -> str:
        """Clean the email body content by removing noise patterns"""
        if not body or pd.isna(body):
            return ""

        cleaned_body = body

        # Remove forwarding headers and chains
        for pattern in self.forwarding_patterns:
            cleaned_body = re.sub(pattern, '', cleaned_body, flags=re.IGNORECASE | re.DOTALL)

        # Remove email signatures
        for pattern in self.signature_patterns:
            cleaned_body = re.sub(pattern, '', cleaned_body, flags=re.IGNORECASE | re.DOTALL)

        # Remove technical artifacts
        for pattern in self.technical_patterns:
            cleaned_body = re.sub(pattern, '', cleaned_body, flags=re.IGNORECASE)

        # Remove attachment references
        for pattern in self.attachment_patterns:
            cleaned_body = re.sub(pattern, '', cleaned_body, flags=re.IGNORECASE)

        # Clean up whitespace and formatting
        cleaned_body = self._clean_whitespace(cleaned_body)

        # Remove very short or meaningless content
        if len(cleaned_body.strip()) < 10:
            return ""

        return cleaned_body.strip()

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace and formatting issues"""
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing spaces
        text = re.sub(r' +\n', '\n', text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)

        return text

    def clean_subject_line(self, subject: str) -> str:
        """Clean subject line"""
        if not subject or pd.isna(subject):
            return ""

        # Remove common prefixes
        subject = re.sub(r'^(RE:|FW:|FWD:)\s*', '', subject, flags=re.IGNORECASE)

        # Remove encoding artifacts
        subject = re.sub(r'=\?.*?\?=', '', subject)

        return subject.strip()

    def standardize_date(self, date_str: str) -> str:
        """Standardize date format"""
        if not date_str or pd.isna(date_str):
            return ""

        try:
            # Try to parse common date formats
            date_formats = [
                "%d.%m.%Y %H:%M:%S",
                "%d.%m.%Y %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M"
            ]

            for fmt in date_formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

            return date_str  # Return original if parsing fails
        except:
            return date_str

    def extract_clean_emails(self, email_field: str) -> List[str]:
        """Extract and clean email addresses"""
        if not email_field or pd.isna(email_field):
            return []

        # Find all email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, email_field)

        # Clean and deduplicate
        clean_emails = []
        for email in emails:
            email = email.lower().strip()
            if email and email not in clean_emails:
                clean_emails.append(email)

        return clean_emails

    def filter_meaningful_content(self, body: str) -> bool:
        """Check if the email body contains meaningful content"""
        if not body or len(body.strip()) < 20:
            return False

        # Check for meaningful words (not just technical jargon)
        word_count = len(body.split())
        if word_count < 5:
            return False

        # Check for common spam indicators
        spam_indicators = [
            'click here', 'free money', 'urgent', 'act now',
            'limited time', 'congratulations'
        ]

        body_lower = body.lower()
        spam_count = sum(1 for indicator in spam_indicators if indicator in body_lower)

        if spam_count > 2:
            return False

        return True

    def clean_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean the entire email dataset"""
        cleaned_data = []

        for email in data:
            try:
                # Clean email body
                cleaned_body = self.clean_email_body(email.get('Body', ''))

                # Skip emails with no meaningful content
                if not self.filter_meaningful_content(cleaned_body):
                    continue

                # Create cleaned email record
                cleaned_email = {
                    'message_id': email.get('Message-ID', '').strip('<>'),
                    'date': self.standardize_date(email.get('Date', '')),
                    'from': self.extract_clean_emails(email.get('From', '')),
                    'to': self.extract_clean_emails(email.get('To', '')),
                    'subject': self.clean_subject_line(email.get('Subject', '')),
                    'body': cleaned_body,
                    'source_file': email.get('SourceFile', ''),
                    'original_length': len(email.get('Body', '')),
                    'cleaned_length': len(cleaned_body)
                }

                cleaned_data.append(cleaned_email)

            except Exception as e:
                print(f"Error processing email {email.get('Message-ID', 'Unknown')}: {str(e)}")
                continue

        return cleaned_data

    def save_cleaned_data(self, cleaned_data: List[Dict[str, Any]],
                          output_file: str = 'cleaned_emails.json'):
        """Save cleaned data to JSON file with proper formatting"""
        # Ensure we're saving as JSON with proper structure
        json_output = {
            "metadata": {
                "total_emails": len(cleaned_data),
                "cleaning_timestamp": datetime.now().isoformat(),
                "data_format": "cleaned_email_dataset",
                "schema_version": "1.0"
            },
            "emails": cleaned_data
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        print(f"✅ Cleaned data saved to {output_file} (JSON format)")
        print(f"📧 Total emails processed: {len(cleaned_data)}")
        print(f"📄 Output format: JSON with metadata")

        return json_output

    def create_json_output_structure(self, cleaned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create the standardized JSON output structure"""
        return {
            "metadata": {
                "total_emails": len(cleaned_data),
                "cleaning_timestamp": datetime.now().isoformat(),
                "data_format": "cleaned_email_dataset",
                "schema_version": "1.0",
                "fields": {
                    "message_id": "Unique identifier for the email",
                    "date": "Standardized timestamp (YYYY-MM-DD HH:MM:SS)",
                    "from": "List of sender email addresses",
                    "to": "List of recipient email addresses",
                    "subject": "Cleaned subject line",
                    "body": "Cleaned email body content",
                    "source_file": "Original source file reference",
                    "original_length": "Character count before cleaning",
                    "cleaned_length": "Character count after cleaning"
                }
            },
            "emails": cleaned_data
        }

    def save_as_json_only(self, cleaned_data: List[Dict[str, Any]],
                          output_file: str = 'cleaned_emails_final.json'):
        """Primary method to save data as JSON with complete structure"""
        json_structure = self.create_json_output_structure(cleaned_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)

        print(f"🎯 PRIMARY OUTPUT: {output_file} (JSON format)")
        print(f"📊 Structure: metadata + {len(cleaned_data)} cleaned emails")

        return json_structure

    def generate_cleaning_report(self, original_data: List[Dict[str, Any]],
                                 cleaned_data: List[Dict[str, Any]]):
        """Generate a report of the cleaning process"""
        report = {
            'original_count': len(original_data),
            'cleaned_count': len(cleaned_data),
            'removed_count': len(original_data) - len(cleaned_data),
            'average_original_length': sum(len(email.get('Body', '')) for email in original_data) / len(original_data),
            'average_cleaned_length': sum(email['cleaned_length'] for email in cleaned_data) / len(
                cleaned_data) if cleaned_data else 0,
            'compression_ratio': 0
        }

        if report['average_original_length'] > 0:
            report['compression_ratio'] = 1 - (report['average_cleaned_length'] / report['average_original_length'])

        print("\n" + "=" * 50)
        print("EMAIL CLEANING REPORT")
        print("=" * 50)
        print(f"Original emails: {report['original_count']}")
        print(f"Cleaned emails: {report['cleaned_count']}")
        print(f"Removed emails: {report['removed_count']}")
        print(f"Average original length: {report['average_original_length']:.0f} characters")
        print(f"Average cleaned length: {report['average_cleaned_length']:.0f} characters")
        print(f"Content reduction: {report['compression_ratio']:.1%}")
        print("=" * 50)

        return report


def main():
    # Initialize the cleaner
    cleaner = EmailDataCleaner()

    # Load the JSON data
    try:
        with open('cleaned_json_50.json', 'r', encoding='utf-8') as f:
            email_data = json.load(f)
        print(f"📥 Loaded {len(email_data)} emails from JSON file")
    except FileNotFoundError:
        print("❌ Error: 'cleaned_json_50.json' file not found")
        return
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format")
        return

    # Clean the data
    print("🧹 Starting email cleaning process...")
    cleaned_emails = cleaner.clean_dataset(email_data)

    # Generate report
    cleaner.generate_cleaning_report(email_data, cleaned_emails)

    # Save as JSON (PRIMARY OUTPUT)
    print("\n" + "=" * 50)
    print("🎯 SAVING AS JSON FORMAT")
    print("=" * 50)

    json_output = cleaner.save_as_json_only(cleaned_emails, 'cleaned_emails_final.json')

    # Show JSON structure preview
    if cleaned_emails:
        print(f"\n📋 JSON OUTPUT STRUCTURE:")
        print("=" * 50)

        # Create a sample with first email
        sample_structure = {
            "metadata": json_output["metadata"],
            "emails": [
                {
                    "message_id": cleaned_emails[0]["message_id"],
                    "date": cleaned_emails[0]["date"],
                    "from": cleaned_emails[0]["from"],
                    "to": cleaned_emails[0]["to"][:2] if len(cleaned_emails[0]["to"]) > 2 else cleaned_emails[0]["to"],
                    # Show max 2 recipients
                    "subject": cleaned_emails[0]["subject"],
                    "body": cleaned_emails[0]["body"][:150] + "..." if len(cleaned_emails[0]["body"]) > 150 else
                    cleaned_emails[0]["body"],
                    "source_file": cleaned_emails[0]["source_file"],
                    "original_length": cleaned_emails[0]["original_length"],
                    "cleaned_length": cleaned_emails[0]["cleaned_length"]
                }
            ]
        }

        print(json.dumps(sample_structure, indent=2, ensure_ascii=False))
        print("=" * 50)
        print(f"✅ COMPLETE DATASET: 'cleaned_emails_final.json'")
        print(f"📧 Contains {len(cleaned_emails)} fully cleaned emails")
        print(f"🏗️  JSON Structure: metadata + emails array")

        # Show schema information
        print(f"\n📝 JSON SCHEMA:")
        for field, description in json_output["metadata"]["fields"].items():
            print(f"  • {field}: {description}")


def example_usage():
    """Example of how to load and use the cleaned JSON data"""
    print("\n" + "=" * 50)
    print("📖 EXAMPLE: How to load the cleaned JSON data")
    print("=" * 50)

    example_code = '''
# Load the cleaned JSON data
import json

with open('cleaned_emails_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Access metadata
print(f"Total emails: {data['metadata']['total_emails']}")
print(f"Cleaned on: {data['metadata']['cleaning_timestamp']}")

# Access emails
emails = data['emails']
for email in emails[:3]:  # First 3 emails
    print(f"Subject: {email['subject']}")
    print(f"From: {email['from']}")
    print(f"Body length: {email['cleaned_length']} chars")
    print("---")
'''
    print(example_code)


if __name__ == "__main__":
    main()

    # Show usage example
    try:
        example_usage()
    except:
        pass