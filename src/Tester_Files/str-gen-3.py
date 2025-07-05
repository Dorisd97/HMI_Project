import json
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Set
import logging

from src.config.config import PROCESSED_JSON_OUTPUT, PROCESSED_JSON_OUTPUT_100

# LangChain and Ollama imports - using compatible versions
try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain.chains.llm import LLMChain

except ImportError:
    # Fallback for different LangChain versions
    try:
        from langchain.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains.llm import LLMChain

    except ImportError:
        print("Warning: LangChain not available. Some features will be disabled.")
        Ollama = None
        PromptTemplate = None
        LLMChain = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailRelationshipAnalyzer:
    def __init__(self, json_file_path: str, model_name: str = "mistral"):
        """
        Initialize the analyzer with email data and LLM model
        """
        self.json_file_path = json_file_path
        self.model_name = model_name

        # Initialize LLM if available
        if Ollama is not None:
            try:
                self.llm = Ollama(model=model_name, temperature=0.3)
                self.llm_available = True
            except Exception as e:
                logger.warning(f"Could not initialize Ollama: {e}")
                self.llm = None
                self.llm_available = False
        else:
            self.llm = None
            self.llm_available = False

        self.relationships = defaultdict(list)
        self.entity_networks = {}
        self.timeline_events = []

        # Load and process data
        self._load_data()

    def _load_data(self):
        """Load and preprocess email data"""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            self.emails_data = data.get('emails', [])
            logger.info(f"Loaded {len(self.emails_data)} emails")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def extract_temporal_relationships(self) -> List[Dict]:
        """Extract temporal relationships between emails"""
        temporal_events = []

        for email in self.emails_data:
            date_str = email.get('date', '')
            try:
                # Parse different date formats
                if '.' in date_str:
                    date_obj = datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
                else:
                    date_obj = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')

                temporal_events.append({
                    'email_id': email['email_id'],
                    'date': date_obj,
                    'subject': email['subject'],
                    'classification': email['classification'],
                    'entities': email['entities'],
                    'tone': email['tone_analysis']
                })
            except:
                logger.warning(f"Could not parse date for email {email['email_id']}: {date_str}")

        # Sort by date
        temporal_events.sort(key=lambda x: x['date'])
        self.timeline_events = temporal_events
        return temporal_events

    def find_entity_overlaps(self) -> Dict[str, List]:
        """Find overlapping entities between emails"""
        entity_overlaps = defaultdict(list)

        for i, email1 in enumerate(self.emails_data):
            for j, email2 in enumerate(self.emails_data[i + 1:], i + 1):
                overlaps = self._calculate_entity_overlap(email1['entities'], email2['entities'])
                if overlaps:
                    entity_overlaps[f"{email1['email_id']}-{email2['email_id']}"] = overlaps

        return dict(entity_overlaps)

    def _calculate_entity_overlap(self, entities1: Dict, entities2: Dict) -> Dict:
        """Calculate overlap between two entity dictionaries"""
        overlaps = {}

        for entity_type in entities1.keys():
            if entity_type in entities2:
                set1 = set(entities1[entity_type])
                set2 = set(entities2[entity_type])
                intersection = set1.intersection(set2)
                if intersection:
                    overlaps[entity_type] = list(intersection)

        return overlaps

    def analyze_communication_patterns(self) -> Dict:
        """Analyze communication patterns between people"""
        communication_graph = nx.DiGraph()

        for email in self.emails_data:
            sender = email.get('from', '')
            recipients = email.get('to', '').split(', ') if email.get('to') else []

            # Clean email addresses
            sender = self._clean_email(sender)
            recipients = [self._clean_email(r) for r in recipients if r]

            # Add edges for communication
            for recipient in recipients:
                if communication_graph.has_edge(sender, recipient):
                    communication_graph[sender][recipient]['weight'] += 1
                else:
                    communication_graph.add_edge(sender, recipient, weight=1, emails=[email['email_id']])
                    communication_graph[sender][recipient]['emails'].append(email['email_id'])

        return communication_graph

    def _clean_email(self, email: str) -> str:
        """Clean and extract domain from email"""
        if '@' in email:
            return email.split('@')[0].replace('.', '_')
        return email

    def analyze_topic_progression(self) -> Dict:
        """Analyze how topics evolve over time using LLM"""
        if not self.llm_available:
            logger.warning("LLM not available. Performing basic topic analysis instead.")
            return self._basic_topic_analysis()

        # Simple prompt without complex template
        emails_context = self._prepare_emails_context()

        prompt = f"""
        Analyze the following email data and identify key topic progressions and relationships:

        {emails_context}

        Please provide:
        1. Main topics/themes that appear across multiple emails
        2. How these topics evolve over time
        3. Key connections between emails based on content
        4. Important business events or decisions mentioned
        5. Power dynamics and relationships revealed

        Format your response as a structured analysis with clear sections.
        """

        try:
            result = self.llm.invoke(prompt)
            return self._parse_llm_analysis(result)
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._basic_topic_analysis()

    def _basic_topic_analysis(self) -> Dict:
        """Basic topic analysis without LLM"""
        analysis = {
            'raw_analysis': 'Basic analysis performed without LLM',
            'topics': [],
            'connections': [],
            'timeline': [],
            'key_insights': []
        }

        # Extract common themes from subjects and summaries
        all_text = []
        for email in self.emails_data:
            text = f"{email.get('subject', '')} {email.get('summary', '')}"
            all_text.append(text.lower())

        # Find common words and phrases
        word_counts = Counter()
        for text in all_text:
            words = text.split()
            for word in words:
                if len(word) > 4:  # Filter out short words
                    word_counts[word] += 1

        # Get top topics
        top_words = word_counts.most_common(20)
        analysis['topics'] = [f"Topic: {word} (mentioned {count} times)"
                              for word, count in top_words]

        # Basic timeline analysis
        dates = []
        for email in self.emails_data:
            date_obj = self._parse_date_safe(email.get('date', ''))
            if date_obj != datetime.min:
                dates.append({
                    'date': date_obj,
                    'subject': email.get('subject', ''),
                    'classification': email.get('classification', '')
                })

        dates.sort(key=lambda x: x['date'])
        analysis['timeline'] = [f"{item['date'].strftime('%Y-%m-%d')}: {item['subject']}"
                                for item in dates[:10]]

        return analysis

    def _prepare_emails_context(self, max_emails: int = 20) -> str:
        """Prepare email context for LLM analysis"""
        context_parts = []

        # Sort emails by date and take a sample
        sorted_emails = sorted(self.emails_data,
                               key=lambda x: self._parse_date_safe(x.get('date', '')))

        sample_emails = sorted_emails[:max_emails]

        for email in sample_emails:
            context_parts.append(f"""
Email ID: {email['email_id']}
Date: {email.get('date', 'Unknown')}
From: {email.get('from', 'Unknown')}
To: {email.get('to', 'Unknown')}
Subject: {email.get('subject', 'No subject')}
Classification: {email.get('classification', 'Unknown')}
Tone: {email.get('tone_analysis', 'Unknown')}
Summary: {email.get('summary', 'No summary')}
Key Entities: {email.get('entities', {})}
---
""")

        return '\n'.join(context_parts)

    def _parse_date_safe(self, date_str: str) -> datetime:
        """Safely parse date string"""
        try:
            if '.' in date_str:
                return datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
            else:
                return datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
        except:
            return datetime.min

    def _parse_llm_analysis(self, llm_result: str) -> Dict:
        """Parse LLM analysis result into structured format"""
        # Basic parsing - can be enhanced
        analysis = {
            'raw_analysis': llm_result,
            'topics': [],
            'connections': [],
            'timeline': [],
            'key_insights': []
        }

        # Simple keyword extraction for topics
        lines = llm_result.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if 'topic' in line.lower() or 'theme' in line.lower():
                current_section = 'topics'
            elif 'connection' in line.lower() or 'relationship' in line.lower():
                current_section = 'connections'
            elif 'timeline' in line.lower() or 'evolution' in line.lower():
                current_section = 'timeline'
            elif line and current_section:
                analysis[current_section].append(line)

        return analysis

    def build_comprehensive_network(self) -> nx.Graph:
        """Build a comprehensive network of all relationships"""
        G = nx.Graph()

        # Add nodes for each email
        for email in self.emails_data:
            G.add_node(f"email_{email['email_id']}",
                       type='email',
                       date=email.get('date', ''),
                       subject=email.get('subject', ''),
                       classification=email.get('classification', ''),
                       tone=email.get('tone_analysis', ''))

        # Add entity nodes and connections
        for email in self.emails_data:
            email_node = f"email_{email['email_id']}"
            entities = email.get('entities', {})

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_node = f"{entity_type}_{entity}"
                    G.add_node(entity_node, type=entity_type, name=entity)
                    G.add_edge(email_node, entity_node, relationship='mentions')

        # Add temporal connections
        sorted_emails = sorted(self.emails_data,
                               key=lambda x: self._parse_date_safe(x.get('date', '')))

        for i in range(len(sorted_emails) - 1):
            current_email = sorted_emails[i]
            next_email = sorted_emails[i + 1]

            # Check for entity overlaps
            overlaps = self._calculate_entity_overlap(
                current_email.get('entities', {}),
                next_email.get('entities', {})
            )

            if overlaps:
                G.add_edge(f"email_{current_email['email_id']}",
                           f"email_{next_email['email_id']}",
                           relationship='temporal_entity_overlap',
                           overlaps=overlaps)

        return G

    def generate_relationship_report(self) -> Dict:
        """Generate comprehensive relationship analysis report"""
        logger.info("Starting comprehensive relationship analysis...")

        report = {
            'temporal_analysis': self.extract_temporal_relationships(),
            'entity_overlaps': self.find_entity_overlaps(),
            'communication_patterns': self.analyze_communication_patterns(),
            'topic_progression': self.analyze_topic_progression(),
            'network_graph': self.build_comprehensive_network(),
            'summary_stats': self._generate_summary_stats()
        }

        # Save report
        self._save_report(report)

        return report

    def _generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        stats = {
            'total_emails': len(self.emails_data),
            'date_range': {},
            'classification_counts': Counter(),
            'tone_counts': Counter(),
            'entity_counts': defaultdict(int),
            'key_players': Counter()
        }

        dates = []
        for email in self.emails_data:
            # Classification and tone counts
            stats['classification_counts'][email.get('classification', 'Unknown')] += 1
            stats['tone_counts'][email.get('tone_analysis', 'Unknown')] += 1

            # Entity counts
            entities = email.get('entities', {})
            for entity_type, entity_list in entities.items():
                stats['entity_counts'][entity_type] += len(entity_list)
                if entity_type == 'people':
                    for person in entity_list:
                        stats['key_players'][person] += 1

            # Date tracking
            date_obj = self._parse_date_safe(email.get('date', ''))
            if date_obj != datetime.min:
                dates.append(date_obj)

        if dates:
            stats['date_range'] = {
                'start': min(dates).isoformat(),
                'end': max(dates).isoformat()
            }

        return stats

    def _save_report(self, report: Dict):
        """Save analysis report to files"""
        # Convert NetworkX graphs to JSON-serializable format
        serializable_report = report.copy()

        # Handle NetworkX graphs
        if 'communication_patterns' in report:
            comm_graph = report['communication_patterns']
            serializable_report['communication_patterns'] = {
                'nodes': list(comm_graph.nodes(data=True)),
                'edges': list(comm_graph.edges(data=True))
            }

        if 'network_graph' in report:
            network_graph = report['network_graph']
            serializable_report['network_graph'] = {
                'nodes': list(network_graph.nodes(data=True)),
                'edges': list(network_graph.edges(data=True))
            }

        # Save to JSON
        output_file = 'email_relationship_analysis.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = EmailRelationshipAnalyzer(PROCESSED_JSON_OUTPUT_100)

    # Generate comprehensive analysis
    report = analyzer.generate_relationship_report()

    print("Analysis complete! Check 'email_relationship_analysis.json' for results.")
    print(f"Found {len(report['entity_overlaps'])} entity overlap relationships")
    print(f"Analyzed {report['summary_stats']['total_emails']} emails")
    print(f"Date range: {report['summary_stats']['date_range']}")


if __name__ == "__main__":
    main()