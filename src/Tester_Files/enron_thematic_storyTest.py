import json
import requests
from typing import Dict, List, Any
from collections import defaultdict
import re
from datetime import datetime
from src.config.config import PROCESSED_JSON_OUTPUT_100, GENERATED_THEME_STORY_PATH2

# ====== CONFIGURATION ======
INPUT_FILE_PATH = PROCESSED_JSON_OUTPUT_100  # Input file path
OUTPUT_FILE_PATH = GENERATED_THEME_STORY_PATH2  # Output file path
OLLAMA_URL = "http://localhost:11434"  # Ollama server URL
MODEL_NAME = "mistral"  # Model to use


class EnronThematicAnalyzer:
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = MODEL_NAME):
        """
        Initialize the analyzer with Ollama server URL and model

        Args:
            ollama_url: URL of the Ollama server
            model: Name of the model to use
        """
        self.ollama_url = ollama_url
        self.model = model

    def call_ollama(self, prompt: str) -> str:
        """
        Make a request to Ollama API

        Args:
            prompt: The prompt to send to the model

        Returns:
            Response text from the model
        """
        try:
            print("Calling Ollama API...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"

    def extract_emails_data(self, json_data: Dict) -> List[Dict]:
        """
        Extract email data from the JSON structure

        Args:
            json_data: The loaded JSON data

        Returns:
            List of email dictionaries
        """
        if "emails" in json_data:
            return json_data["emails"]
        else:
            raise ValueError("JSON data does not contain 'emails' key")

    def identify_themes(self, emails: List[Dict]) -> Dict[str, Any]:
        """
        Identify major themes across all emails

        Args:
            emails: List of email dictionaries

        Returns:
            Dictionary containing identified themes
        """
        print("Identifying themes...")

        # Prepare email summaries for analysis
        email_summaries = []
        for email in emails:
            email_info = {
                "id": email.get("email_id"),
                "date": email.get("date"),
                "subject": email.get("subject", ""),
                "summary": email.get("summary", ""),
                "classification": email.get("classification", ""),
                "entities": email.get("entities", {}),
                "tone": email.get("tone_analysis", "")
            }
            email_summaries.append(email_info)

        prompt = f"""
        Analyze the following Enron email dataset and identify the major thematic stories present. 
        Look for recurring patterns, narratives, and connections between emails.

        Be sure to identify the following themes and provide detailed explanations:

        1. The Dynegy Merger Saga
           - Look for initial merger announcements, discussions about leadership transitions, technical details about stock exchanges, and the eventual collapse of the merger deal.
           - What were the key turning points that led to the failure of this merger?

        2. California Energy Crisis and Market Manipulation
           - Look for references to the California energy crisis, such as Governor Gray Davis' demand for refunds, price manipulation investigations, and FERC interventions.
           - How did Enron attempt to manipulate energy prices during this time?

        3. Financial Engineering and Special Purpose Entities
           - Identify emails referencing special purpose entities, off-balance-sheet transactions, and LJM partnerships.
           - Look for financial strategies that were used to hide debt and manipulate financial reports.

        4. The Unraveling and Bankruptcy
           - Trace the sequence of events that led to Enron's bankruptcy filing, starting from early signs of financial distress to the eventual Chapter 11 filing.
           - Look for emails discussing SEC investigations, asset auctions, and creditor negotiations.

        5. Regulatory and Legal Battles
           - Identify emails mentioning FERC proceedings, subpoenas, and legal challenges Enron faced.
           - What were the major legal battles that played a role in Enron's collapse?

        6. Energy Trading Operations
           - Look for discussions on Enron's energy trading operations, risk management, and mark-to-market accounting practices.
           - What were the key events related to energy trading that affected Enron's financial health?

        7. Corporate Culture and Internal Dynamics
           - Look for emails that describe internal office politics, restructuring efforts, and the company's corporate culture.
           - How did these internal dynamics contribute to Enron's downfall?

        Please provide a detailed description of each theme, along with key characteristics, references to specific emails, and an explanation of how they relate to the overall Enron story.
        """

        response = self.call_ollama(prompt)
        return {"themes_analysis": response, "email_summaries": email_summaries}

    def map_emails_to_themes(self, emails: List[Dict], themes_analysis: str) -> Dict[str, Any]:
        """
        Map specific emails to identified themes

        Args:
            emails: List of email dictionaries
            themes_analysis: The themes analysis from previous step

        Returns:
            Dictionary with theme mappings and explanations
        """
        print("Mapping emails to themes...")

        email_details = []
        for email in emails:
            email_detail = {
                "id": email.get("email_id"),
                "date": email.get("date"),
                "subject": email.get("subject", ""),
                "summary": email.get("summary", "")[:200] + "..." if len(email.get("summary", "")) > 200 else email.get(
                    "summary", ""),
                "classification": email.get("classification", ""),
                "key_entities": {
                    "organizations": email.get("entities", {}).get("organizations", [])[:3],
                    "people": email.get("entities", {}).get("people", [])[:3],
                    "projects": email.get("entities", {}).get("projects", [])[:3]
                }
            }
            email_details.append(email_detail)

        prompt = f"""
        Map every email in the provided dataset to one of the identified themes from the previous analysis. 

        Use the following themes:
        1. Dynegy Merger Saga
        2. California Energy Crisis and Market Manipulation
        3. Financial Engineering and Special Purpose Entities
        4. The Unraveling and Bankruptcy
        5. Regulatory and Legal Battles
        6. Energy Trading Operations
        7. Corporate Culture and Internal Dynamics

        For each email, please provide:
        - A clear identification of the theme(s) that the email belongs to.
        - A brief explanation of why this email is relevant to the identified theme(s).
        """

        response = self.call_ollama(prompt)
        return self._parse_theme_mapping_response(response)

    def _parse_theme_mapping_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the theme mapping response

        Args:
            response: Raw response from the model

        Returns:
            Parsed theme mapping dictionary
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass

        # Fallback parsing
        theme_mapping = {"theme_mappings": {}}
        lines = response.split('\n')
        current_theme = None

        for line in lines:
            line = line.strip()
            if 'theme' in line.lower() and ':' in line:
                current_theme = line.split(':', 1)[1].strip().strip('"')
                theme_mapping["theme_mappings"][current_theme] = {"email_ids": [], "explanation": ""}
            elif 'email' in line.lower() and current_theme:
                email_ids = re.findall(r'\d+', line)
                theme_mapping["theme_mappings"][current_theme]["email_ids"] = [int(id_) for id_ in email_ids]
            elif 'explanation' in line.lower() and current_theme:
                theme_mapping["theme_mappings"][current_theme]["explanation"] = line.split(':', 1)[1].strip()

        return theme_mapping

    def generate_narrative_timeline(self, emails: List[Dict], theme_mapping: Dict[str, Any]) -> str:
        """
        Generate a chronological narrative timeline

        Args:
            emails: List of email dictionaries
            theme_mapping: Mapping of themes to email IDs

        Returns:
            Narrative timeline string
        """
        print("Generating narrative timeline...")

        # Sort emails by date
        try:
            sorted_emails = sorted(emails, key=lambda x: x.get("date", ""))
        except:
            sorted_emails = emails

        # Create timeline data
        timeline_data = []
        for email in sorted_emails[:30]:  # Limit for token constraints
            timeline_data.append({
                "id": email.get("email_id"),
                "date": email.get("date"),
                "subject": email.get("subject", ""),
                "summary": email.get("summary", "")[:150] + "..." if len(email.get("summary", "")) > 150 else email.get(
                    "summary", ""),
                "classification": email.get("classification", "")
            })

        prompt = f"""
        Generate a chronological narrative timeline based on the provided email data and themes.

        Ensure the timeline is broken down into distinct phases:
        1. Early Operations (normal business)
        2. Growing Problems (regulatory issues, investigations)
        3. Crisis Phase (mergers, financial troubles)
        4. Collapse (bankruptcy, legal consequences)

        For each phase, provide:
        - Key emails from the dataset that represent the phase.
        - A brief description of the events happening during this phase.
        - The connection to the broader narrative, showing how the emails build upon each other.

        Please provide the narrative in a structured format, detailing how Enron's situation evolved from normal business operations to eventual bankruptcy.
        """

        return self.call_ollama(prompt)

    def analyze_dataset(self, json_file_path: str) -> Dict[str, Any]:
        """
        Complete analysis pipeline

        Args:
            json_file_path: Path to the JSON file containing email data

        Returns:
            Complete analysis results
        """
        print(f"Loading email data from: {json_file_path}")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {json_file_path}")

        emails = self.extract_emails_data(json_data)
        print(f"Loaded {len(emails)} emails")

        # Step 1: Identify themes
        themes_result = self.identify_themes(emails)

        # Step 2: Map emails to themes
        theme_mapping = self.map_emails_to_themes(emails, themes_result["themes_analysis"])

        # Step 3: Generate narrative timeline
        narrative_timeline = self.generate_narrative_timeline(emails, theme_mapping)

        # Compile results
        results = {
            "analysis_metadata": {
                "total_emails": len(emails),
                "analysis_timestamp": datetime.now().isoformat(),
                "input_file": json_file_path,
                "model_used": self.model
            },
            "themes_analysis": themes_result["themes_analysis"],
            "theme_mapping": theme_mapping,
            "narrative_timeline": narrative_timeline,
            "email_summaries": themes_result["email_summaries"]
        }

        return results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save analysis results to JSON file

        Args:
            results: Analysis results dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the analysis results

        Args:
            results: Analysis results dictionary
        """
        print("\n" + "=" * 80)
        print("ENRON EMAIL THEMATIC ANALYSIS SUMMARY")
        print("=" * 80)

        metadata = results.get('analysis_metadata', {})
        print(f"Total Emails Analyzed: {metadata.get('total_emails', 'Unknown')}")
        print(f"Analysis Timestamp: {metadata.get('analysis_timestamp', 'Unknown')}")
        print(f"Model Used: {metadata.get('model_used', 'Unknown')}")

        print("\nIDENTIFIED THEMES:")
        print("-" * 40)
        print(results.get('themes_analysis', 'No themes analysis available'))

        theme_mapping = results.get('theme_mapping', {}).get('theme_mappings', {})
        if theme_mapping:
            print("\nTHEME TO EMAIL MAPPING:")
            print("-" * 40)
            for theme, details in theme_mapping.items():
                email_ids = details.get('email_ids', [])
                explanation = details.get('explanation', 'No explanation')
                print(f"\n{theme}:")
                print(f"  Emails: {email_ids}")
                print(f"  Explanation: {explanation}")

        print("\nNARRATIVE TIMELINE:")
        print("-" * 40)
        print(results.get('narrative_timeline', 'No timeline available'))


def main():
    """
    Main function to run the analysis
    """
    print("Starting Enron Email Thematic Analysis...")
    print(f"Input file: {INPUT_FILE_PATH}")
    print(f"Output file: {OUTPUT_FILE_PATH}")
    print(f"Ollama server: {OLLAMA_URL}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 60)

    # Initialize analyzer
    analyzer = EnronThematicAnalyzer()

    try:
        # Run complete analysis
        results = analyzer.analyze_dataset(INPUT_FILE_PATH)

        # Save results to JSON
        analyzer.save_results(results, OUTPUT_FILE_PATH)

        # Print summary
        analyzer.print_summary(results)

        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_FILE_PATH}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
