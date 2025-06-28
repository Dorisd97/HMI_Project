#!/usr/bin/env python3
"""
Enron Email LLM Analysis Processor
This script runs all LLM analyses and saves results to JSON files.
Run this once before using the Streamlit app.
"""

import json
import os
from datetime import datetime
import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import hashlib

from src.config.config import PROCESSED_JSON_OUTPUT_100


class EnronLLMProcessor:
    def __init__(self, email_data_path: str, output_dir: str = "enron_analysis_output"):
        self.email_data_path = email_data_path
        self.output_dir = output_dir
        self.llm = Ollama(model="mistral", temperature=0.7)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load email data
        with open(email_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both formats
        if isinstance(data, list):
            self.emails = data
        elif isinstance(data, dict) and 'emails' in data:
            self.emails = data['emails']
        else:
            raise ValueError("Invalid JSON format")

        self.df = pd.DataFrame(self.emails)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        print(f"Loaded {len(self.emails)} emails")

    def analyze_individual_email(self, email: Dict) -> Dict:
        """Analyze a single email using LLM"""
        prompt = PromptTemplate(
            input_variables=["sender", "recipient", "date", "subject", "summary", "tone", "classification", "entities"],
            template="""
            You are a detective analyzing corporate emails. Examine this email without any preconceptions:

            From: {sender}
            To: {recipient}
            Date: {date}
            Subject: {subject}
            Summary: {summary}
            Tone: {tone}
            Classification: {classification}
            Mentioned Entities: {entities}

            Analyze this email like you're discovering it for the first time:
            1. What seems significant or unusual about this communication?
            2. What relationships or power dynamics are revealed?
            3. Are there any warning signs, red flags, or interesting patterns?
            4. What questions does this email raise?
            5. How might this connect to other events (without assuming what those are)?

            Be investigative and curious. Point out anything noteworthy.
            Keep analysis to 3-4 sentences.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Prepare entities string
        entities = email.get('entities', {})
        entities_str = f"People: {', '.join(entities.get('people', [])[:5])}, " \
                       f"Orgs: {', '.join(entities.get('organizations', [])[:5])}, " \
                       f"Projects: {', '.join(entities.get('projects', [])[:3])}"

        try:
            analysis = chain.run(
                sender=email.get('from', 'Unknown'),
                recipient=email.get('to', 'Unknown'),
                date=email.get('date', 'Unknown'),
                subject=email.get('subject', 'No subject'),
                summary=email.get('summary', 'No summary'),
                tone=email.get('tone_analysis', 'Unknown'),
                classification=email.get('classification', 'Unknown'),
                entities=entities_str
            )

            return {
                'email_id': email.get('email_id'),
                'date': email.get('date'),
                'subject': email.get('subject'),
                'analysis': analysis,
                'status': 'success'
            }
        except Exception as e:
            return {
                'email_id': email.get('email_id'),
                'date': email.get('date'),
                'subject': email.get('subject'),
                'analysis': f"Analysis failed: {str(e)}",
                'status': 'error'
            }

    def extract_key_themes(self) -> List[str]:
        """Let LLM extract key themes from emails"""
        print("Extracting key themes...")

        # Prepare email summaries for analysis
        email_summaries = "\n".join([
            f"- {email.get('date', 'Unknown date')}: {email.get('subject', 'No subject')} - {email.get('summary', '')[:200]}"
            for email in self.emails[:50]  # Sample for theme extraction
        ])

        theme_prompt = PromptTemplate(
            input_variables=["email_summaries"],
            template="""
            Analyze these corporate email summaries and identify the major themes and patterns:

            {email_summaries}

            Extract and list the 10-12 most significant recurring themes, events, or topics.
            Focus on identifying patterns, crises, business relationships, and major developments.
            List them in order of importance, separated by semicolons.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=theme_prompt)

        try:
            themes_text = chain.run(email_summaries=email_summaries)
            themes = [theme.strip() for theme in themes_text.split(';') if theme.strip()]
            return themes[:12]
        except Exception as e:
            print(f"Theme extraction failed: {e}")
            return ["Email analysis failed"]

    def extract_key_events(self) -> List[Dict]:
        """Let LLM identify key events from the emails"""
        print("Identifying key events...")

        # Get emails sorted by date
        sorted_emails = self.df[self.df['date'].notna()].sort_values('date')

        # Prepare data for LLM analysis
        email_timeline = "\n".join([
                                       f"{row['date'].strftime('%Y-%m-%d')}: {row['subject']} ({row['classification']}) - {row['summary'][:150]}..."
                                       for _, row in sorted_emails.iterrows()
                                   ][:60])  # Limit to prevent token overflow

        events_prompt = PromptTemplate(
            input_variables=["email_timeline"],
            template="""
            Analyze this chronological list of corporate emails and identify the most critical events:

            {email_timeline}

            Identify the 8-10 most significant events or turning points based on these emails.
            For each event, provide:
            - Date (YYYY-MM-DD format)
            - Event name (brief description)
            - Significance level (1-3, where 3 is most critical)
            - Brief explanation of why this is significant

            Format each event as: Date|Event|Significance|Explanation
            One event per line.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=events_prompt)

        try:
            events_text = chain.run(email_timeline=email_timeline)
            events = []

            for line in events_text.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        events.append({
                            'date': parts[0].strip(),
                            'event': parts[1].strip(),
                            'significance': float(parts[2].strip()) if parts[2].strip().replace('.',
                                                                                                '').isdigit() else 2,
                            'explanation': parts[3].strip() if len(parts) > 3 else ''
                        })

            return events
        except Exception as e:
            print(f"Event extraction failed: {e}")
            return []

    def analyze_patterns(self, email_analyses: List[Dict]) -> str:
        """Analyze overall patterns from individual analyses"""
        print("Analyzing patterns...")

        pattern_prompt = PromptTemplate(
            input_variables=["email_analyses"],
            template="""
            You are analyzing emails from a major corporation. Based on these individual email analyses, 
            identify the overall story arc and key patterns:

            {email_analyses}

            Provide a comprehensive analysis covering:
            1. The major story arc (beginning, development, crisis points, resolution)
            2. Key players and their roles in the story
            3. Critical turning points and their significance
            4. Cause and effect relationships between events
            5. The human drama and corporate culture elements
            6. Warning signs that were missed or ignored
            7. The broader implications of what you've discovered

            Write a detailed analytical summary that connects all the dots.
            """
        )

        # Prepare email analyses text
        analyses_text = "\n\n".join([
            f"Email {i + 1} ({analysis['date']}): {analysis['subject']}\n"
            f"Analysis: {analysis['analysis']}"
            for i, analysis in enumerate(email_analyses) if analysis['status'] == 'success'
        ])

        chain = LLMChain(llm=self.llm, prompt=pattern_prompt)

        try:
            pattern_analysis = chain.run(email_analyses=analyses_text[:4000])  # Limit tokens
            return pattern_analysis
        except Exception as e:
            return f"Pattern analysis failed: {str(e)}"

    def generate_narrative(self, pattern_analysis: str, key_themes: List[str], key_events: List[Dict]) -> str:
        """Generate the final narrative story"""
        print("Generating narrative...")

        narrative_prompt = PromptTemplate(
            input_variables=["pattern_analysis", "key_themes", "key_events"],
            template="""
            Based on your analysis of corporate emails, create a compelling documentary-style narrative.

            Pattern Analysis:
            {pattern_analysis}

            Key Themes Discovered:
            {key_themes}

            Critical Events Timeline:
            {key_events}

            Write a comprehensive story that:
            1. Tells the complete story discovered in the emails
            2. Structures it like a documentary with clear chapters/acts
            3. Highlights surprises and key discoveries
            4. Connects all events showing cause and effect
            5. Explains the significance of what happened
            6. Captures both business and human elements
            7. Draws lessons from the patterns discovered

            Make it engaging and let the data tell its own story. Write it as if explaining to someone 
            who knows nothing about this company. Use vivid language and storytelling techniques.
            """
        )

        # Format events for prompt
        events_text = "\n".join([
            f"{event['date']}: {event['event']} (Significance: {event['significance']}/3) - {event.get('explanation', '')}"
            for event in key_events
        ])

        chain = LLMChain(llm=self.llm, prompt=narrative_prompt)

        try:
            narrative = chain.run(
                pattern_analysis=pattern_analysis,
                key_themes="; ".join(key_themes),
                key_events=events_text
            )
            return narrative
        except Exception as e:
            return f"Narrative generation failed: {str(e)}"

    def process_all(self):
        """Run all analyses and save results"""
        print("Starting comprehensive email analysis...")
        start_time = datetime.now()

        # 1. Analyze individual emails
        print("\n1. Analyzing individual emails...")
        email_analyses = []

        # Select important emails for analysis
        important_emails = self.select_important_emails(50)

        for email in tqdm(important_emails, desc="Analyzing emails"):
            analysis = self.analyze_individual_email(email)
            email_analyses.append(analysis)

        # Save email analyses
        with open(os.path.join(self.output_dir, 'email_analyses.json'), 'w', encoding='utf-8') as f:
            json.dump(
                [{**e, "date": str(e["date"]) if isinstance(e.get("date"), pd.Timestamp) else e.get("date")}
                 for e in email_analyses],
                f, indent=2, ensure_ascii=False
            )

        # 2. Extract themes
        print("\n2. Extracting key themes...")
        themes = self.extract_key_themes()

        # 3. Extract key events
        print("\n3. Identifying key events...")
        events = self.extract_key_events()

        # 4. Analyze patterns
        print("\n4. Analyzing patterns...")
        pattern_analysis = self.analyze_patterns(email_analyses)

        # 5. Generate narrative
        print("\n5. Generating narrative...")
        narrative = self.generate_narrative(pattern_analysis, themes, events)

        # 6. Prepare complete analysis
        complete_analysis = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'total_emails': len(self.emails),
                'emails_analyzed': len(email_analyses),
                'processing_time': str(datetime.now() - start_time),
                'model_used': 'mistral',
                'version': '1.0'
            },
            'themes': themes,
            'key_events': events,
            'pattern_analysis': pattern_analysis,
            'narrative': narrative,
            'email_analyses': [
            {**e, "date": str(e["date"]) if isinstance(e.get("date"), pd.Timestamp) else e.get("date")}
                for e in email_analyses]
        }

        # Save complete analysis
        with open(os.path.join(self.output_dir, 'complete_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(complete_analysis, f, indent=2, ensure_ascii=False)

        # Also save individual components for easy access
        with open(os.path.join(self.output_dir, 'themes.json'), 'w', encoding='utf-8') as f:
            json.dump(themes, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.output_dir, 'key_events.json'), 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.output_dir, 'narrative.txt'), 'w', encoding='utf-8') as f:
            f.write(narrative)

        print(f"\n✅ Analysis complete! Time taken: {datetime.now() - start_time}")
        print(f"Results saved to: {self.output_dir}")

        return complete_analysis

    def select_important_emails(self, count: int) -> List[Dict]:
        """Select the most important emails for analysis"""
        important_emails = []

        # Priority 1: Critical period (Nov-Dec 2001 if present)
        critical_period = self.df[
            (self.df['date'] >= '2001-11-01') &
            (self.df['date'] <= '2001-12-31')
            ]
        if not critical_period.empty:
            important_emails.extend(critical_period.to_dict('records'))

        # Priority 2: Legal/Compliance and Crisis communications
        high_priority = self.df[
            self.df['classification'].isin([
                'Regulatory Alert / Crisis Communication',
                'Legal/Compliance Matter'
            ])
        ]
        if not high_priority.empty:
            important_emails.extend(high_priority.to_dict('records'))

        # Priority 3: Fill remaining with diverse emails
        if len(important_emails) < count:
            remaining = self.df[~self.df.index.isin([e.get('email_id') for e in important_emails])]
            if not remaining.empty:
                # Get diverse sample across time periods
                remaining_sorted = remaining.sort_values('date', ascending=False)
                additional = remaining_sorted.head(count - len(important_emails))
                important_emails.extend(additional.to_dict('records'))

        # Remove duplicates and limit to count
        seen = set()
        unique_emails = []
        for email in important_emails:
            email_id = email.get('email_id')
            if email_id not in seen:
                seen.add(email_id)
                unique_emails.append(email)

        return unique_emails[:count]


def main():
    parser = argparse.ArgumentParser(description='Process Enron emails with LLM analysis')
    parser.add_argument('--input', '-i', default='enron_emails.json',
                        help='Path to input JSON file (default: enron_emails.json)')
    parser.add_argument('--output', '-o', default='enron_analysis_output',
                        help='Output directory (default: enron_analysis_output)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-analysis even if output exists')

    # args = parser.parse_args()
    #
    # # Check if analysis already exists
    # if os.path.exists(os.path.join(args.output, 'complete_analysis.json')) and not args.force:
    #     print(f"Analysis already exists in {args.output}")
    #     print("Use --force to re-run the analysis")
    #     return
    #
    # # Check if input file exists
    # if not os.path.exists(args.input):
    #     print(f"Error: Input file '{args.input}' not found!")
    #     return

    output_dir = "enron_analysis_output"

    # Run processor with config-based path
    processor = EnronLLMProcessor(email_data_path=PROCESSED_JSON_OUTPUT_100, output_dir=output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()