import json
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StoryEvent:
    """Represents a single event in a story timeline"""
    date: datetime
    email_id: int
    subject: str
    summary: str
    from_person: str
    to_person: str
    key_entities: Dict[str, List[str]]
    relevance_score: float
    email_data: Dict[str, Any]


@dataclass
class StoryCard:
    """Represents a complete story with summary and timeline"""
    keyword: str
    title: str
    summary: str
    key_people: List[str]
    key_organizations: List[str]
    timeline: List[StoryEvent]
    date_range: tuple
    total_emails: int
    relevance_scores: List[float]


class StoryGenerator:
    def __init__(self, email_data):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self._prepare_search_corpus()

    def _prepare_search_corpus(self):
        """Prepare search corpus for similarity matching"""
        # Create searchable text for each email
        self.search_texts = []
        for email in self.emails:
            search_text = f"{email.get('subject', '')} {email.get('summary', '')}"
            # Add entities as searchable text
            entities = email.get('entities', {})
            for entity_type, entity_list in entities.items():
                search_text += " " + " ".join(entity_list)
            self.search_texts.append(search_text)

        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)

    def find_relevant_emails(self, keyword: str, similarity_threshold: float = 0.1) -> List[tuple]:
        """Find emails relevant to a keyword with similarity scores"""
        # Transform keyword query
        keyword_vector = self.vectorizer.transform([keyword])

        # Calculate similarity scores
        similarities = cosine_similarity(keyword_vector, self.tfidf_matrix).flatten()

        # Get relevant emails with scores
        relevant_emails = []
        for i, score in enumerate(similarities):
            if score > similarity_threshold:
                relevant_emails.append((i, score, self.emails[i]))

        # Also check for exact keyword matches in entities and text
        keyword_lower = keyword.lower()
        for i, email in enumerate(self.emails):
            # Check entities
            entities = email.get('entities', {})
            entity_match = False
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if keyword_lower in entity.lower():
                        entity_match = True
                        break
                if entity_match:
                    break

            # Check text content
            text_match = (keyword_lower in email.get('subject', '').lower() or
                          keyword_lower in email.get('summary', '').lower())

            if entity_match or text_match:
                # Boost score if not already included
                existing_indices = [item[0] for item in relevant_emails]
                if i not in existing_indices:
                    boost_score = 0.5 if entity_match else 0.3
                    relevant_emails.append((i, boost_score, email))
                else:
                    # Boost existing score
                    for j, (idx, score, email_data) in enumerate(relevant_emails):
                        if idx == i:
                            relevant_emails[j] = (idx, min(1.0, score + 0.3), email_data)

        # Sort by relevance score
        relevant_emails.sort(key=lambda x: x[1], reverse=True)
        return relevant_emails

    def create_story_timeline(self, relevant_emails: List[tuple]) -> List[StoryEvent]:
        """Create chronological timeline from relevant emails"""
        events = []

        for email_idx, relevance_score, email in relevant_emails:
            try:
                date = datetime.strptime(email['date'], '%d.%m.%Y %H:%M:%S')

                event = StoryEvent(
                    date=date,
                    email_id=email['email_id'],
                    subject=email.get('subject', 'No Subject'),
                    summary=email.get('summary', ''),
                    from_person=email.get('from', ''),
                    to_person=email.get('to', ''),
                    key_entities=email.get('entities', {}),
                    relevance_score=relevance_score,
                    email_data=email
                )
                events.append(event)
            except ValueError:
                continue  # Skip emails with invalid dates

        # Sort chronologically
        events.sort(key=lambda x: x.date)
        return events

    def extract_key_actors(self, events: List[StoryEvent]) -> tuple:
        """Extract key people and organizations from story events"""
        people_counter = Counter()
        org_counter = Counter()

        for event in events:
            # Count email participants
            people_counter[event.from_person] += 1
            people_counter[event.to_person] += 1

            # Count entities
            entities = event.key_entities
            if 'people' in entities:
                for person in entities['people']:
                    people_counter[person] += 2  # Higher weight for mentioned people

            if 'organizations' in entities:
                for org in entities['organizations']:
                    org_counter[org] += 1

        # Get top actors
        key_people = [person for person, count in people_counter.most_common(10) if person]
        key_orgs = [org for org, count in org_counter.most_common(5) if org]

        return key_people, key_orgs

    def generate_story_summary(self, keyword: str, events: List[StoryEvent],
                               key_people: List[str], key_orgs: List[str]) -> str:
        """Generate a narrative summary of the story"""
        if not events:
            return f"No significant email activity found related to '{keyword}'."

        # Basic story statistics
        total_emails = len(events)
        date_range = f"{events[0].date.strftime('%B %Y')} to {events[-1].date.strftime('%B %Y')}"
        duration = (events[-1].date - events[0].date).days

        # Identify story phases based on email density
        story_phases = self._identify_story_phases(events)

        # Generate summary
        summary_parts = []

        # Opening
        summary_parts.append(f"The '{keyword}' story spans {duration} days from {date_range}, "
                             f"involving {total_emails} emails and {len(key_people)} key participants.")

        # Key players
        if key_people:
            top_people = key_people[:3]
            summary_parts.append(f"Primary actors include: {', '.join(top_people)}.")

        if key_orgs:
            summary_parts.append(f"Key organizations involved: {', '.join(key_orgs[:3])}.")

        # Story phases
        if len(story_phases) > 1:
            summary_parts.append(f"The story unfolds in {len(story_phases)} main phases:")
            for i, phase in enumerate(story_phases[:3], 1):
                phase_desc = f"Phase {i} ({phase['start'].strftime('%b %Y')}): {phase['description']}"
                summary_parts.append(phase_desc)

        # Conclusion
        peak_period = max(story_phases, key=lambda x: x['email_count'])
        summary_parts.append(f"Peak activity occurred in {peak_period['start'].strftime('%B %Y')} "
                             f"with {peak_period['email_count']} related emails.")

        return " ".join(summary_parts)

    def _identify_story_phases(self, events: List[StoryEvent]) -> List[Dict]:
        """Identify distinct phases in the story based on email density and content"""
        if not events:
            return []

        # Group events by month
        monthly_groups = defaultdict(list)
        for event in events:
            month_key = event.date.replace(day=1)
            monthly_groups[month_key].append(event)

        # Identify phases based on activity spikes
        phases = []
        sorted_months = sorted(monthly_groups.keys())

        for month in sorted_months:
            month_events = monthly_groups[month]

            # Analyze dominant themes for this month
            classifications = [event.email_data.get('classification', '') for event in month_events]
            dominant_class = Counter(classifications).most_common(1)[0][0] if classifications else "General"

            # Create phase description
            phase_desc = f"{len(month_events)} emails primarily about {dominant_class.lower()}"

            phases.append({
                'start': month,
                'email_count': len(month_events),
                'description': phase_desc,
                'events': month_events
            })

        return phases

    def generate_story_card(self, keyword: str, max_emails: int = 50) -> StoryCard:
        """Generate a complete story card for a given keyword"""
        # Find relevant emails
        relevant_emails = self.find_relevant_emails(keyword)

        # Limit number of emails if too many
        if len(relevant_emails) > max_emails:
            relevant_emails = relevant_emails[:max_emails]

        if not relevant_emails:
            return StoryCard(
                keyword=keyword,
                title=f"No Story Found for '{keyword}'",
                summary=f"No significant email activity found related to '{keyword}'.",
                key_people=[],
                key_organizations=[],
                timeline=[],
                date_range=(None, None),
                total_emails=0,
                relevance_scores=[]
            )

        # Create timeline
        timeline = self.create_story_timeline(relevant_emails)

        # Extract key actors
        key_people, key_orgs = self.extract_key_actors(timeline)

        # Generate summary
        summary = self.generate_story_summary(keyword, timeline, key_people, key_orgs)

        # Create title
        title = f"The {keyword.title()} Story"
        if key_orgs:
            title += f" - {key_orgs[0]}"

        # Date range
        date_range = (timeline[0].date, timeline[-1].date) if timeline else (None, None)

        # Relevance scores
        relevance_scores = [score for _, score, _ in relevant_emails]

        return StoryCard(
            keyword=keyword,
            title=title,
            summary=summary,
            key_people=key_people,
            key_organizations=key_orgs,
            timeline=timeline,
            date_range=date_range,
            total_emails=len(timeline),
            relevance_scores=relevance_scores
        )


class StoryExplorer:
    """Interactive story exploration interface"""

    def __init__(self, email_data):
        self.generator = StoryGenerator(email_data)
        self.story_cache = {}

    def explore_keyword(self, keyword: str) -> StoryCard:
        """Explore a story for a given keyword with caching"""
        if keyword in self.story_cache:
            return self.story_cache[keyword]

        story_card = self.generator.generate_story_card(keyword)
        self.story_cache[keyword] = story_card
        return story_card

    def get_related_keywords(self, keyword: str, limit: int = 5) -> List[str]:
        """Find related keywords/topics that might have interesting stories"""
        story_card = self.explore_keyword(keyword)

        related_keywords = set()

        # Extract from entities in relevant emails
        for event in story_card.timeline:
            entities = event.key_entities
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() != keyword.lower() and len(entity) > 2:
                        related_keywords.add(entity)

        # Add key organizations and people as potential keywords
        related_keywords.update(story_card.key_organizations)
        related_keywords.update([person.split('@')[0] for person in story_card.key_people if '@' in person])

        return list(related_keywords)[:limit]

    def create_story_dashboard(self) -> Dict[str, Any]:
        """Create a dashboard of interesting stories to explore"""
        # Pre-defined interesting keywords for Enron
        interesting_keywords = [
            "merger", "acquisition", "dynegy", "california", "crisis",
            "bankruptcy", "trading", "power", "pipeline", "SEC",
            "investigation", "accounting", "partnerships", "offshore"
        ]

        dashboard = {
            'featured_stories': [],
            'trending_topics': [],
            'timeline_overview': {}
        }

        for keyword in interesting_keywords:
            story_card = self.explore_keyword(keyword)
            if story_card.total_emails > 5:  # Only include substantial stories
                dashboard['featured_stories'].append({
                    'keyword': keyword,
                    'title': story_card.title,
                    'summary': story_card.summary[:200] + "...",
                    'email_count': story_card.total_emails,
                    'date_range': story_card.date_range,
                    'key_people': story_card.key_people[:3]
                })

        # Sort by email count for trending topics
        dashboard['featured_stories'].sort(key=lambda x: x['email_count'], reverse=True)
        dashboard['trending_topics'] = dashboard['featured_stories'][:5]

        return dashboard


def main():
    # Load your JSON data
    with open('enron_emails.json', 'r') as f:
        email_data = json.load(f)

    # Create story explorer
    explorer = StoryExplorer(email_data)

    # Test with sample keywords
    test_keywords = ["merger", "california", "trading"]

    for keyword in test_keywords:
        print(f"\n{'=' * 50}")
        print(f"STORY: {keyword.upper()}")
        print('=' * 50)

        story_card = explorer.explore_keyword(keyword)

        print(f"Title: {story_card.title}")
        print(f"Total Emails: {story_card.total_emails}")
        if story_card.date_range[0]:
            print(
                f"Date Range: {story_card.date_range[0].strftime('%Y-%m-%d')} to {story_card.date_range[1].strftime('%Y-%m-%d')}")
        print(f"Key People: {', '.join(story_card.key_people[:5])}")
        print(f"Key Organizations: {', '.join(story_card.key_organizations[:3])}")
        print(f"\nSummary:\n{story_card.summary}")

        # Show timeline highlights
        if story_card.timeline:
            print(f"\nTimeline Highlights:")
            for i, event in enumerate(story_card.timeline[:5]):
                print(f"{i + 1}. {event.date.strftime('%Y-%m-%d')}: {event.subject[:60]}...")

        # Related keywords
        related = explorer.get_related_keywords(keyword)
        print(f"\nRelated Topics: {', '.join(related)}")

    # Create dashboard
    print(f"\n{'=' * 50}")
    print("STORY DASHBOARD")
    print('=' * 50)

    dashboard = explorer.create_story_dashboard()

    print("Featured Stories:")
    for story in dashboard['featured_stories'][:5]:
        print(f"- {story['title']} ({story['email_count']} emails)")
        print(f"  {story['summary']}")


if __name__ == "__main__":
    main()
