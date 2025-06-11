import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass
from typing import List, Dict, Any
import base64
from io import BytesIO
import plotly.figure_factory as ff

# Configure Streamlit page
st.set_page_config(
    page_title="Enron Story Explorer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .story-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .timeline-item {
        background: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .timeline-item:hover {
        transform: translateX(5px);
    }

    .cluster-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #28a745;
    }

    .keyword-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Data Classes
@dataclass
class StoryEvent:
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
    keyword: str
    title: str
    summary: str
    key_people: List[str]
    key_organizations: List[str]
    timeline: List[StoryEvent]
    date_range: tuple
    total_emails: int
    relevance_scores: List[float]


# Core Analysis Classes
class EmailThreadAnalyzer:
    """Analyze email threads and conversation patterns"""

    def __init__(self, email_data):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)
        self.threads = {}

    def clean_subject(self, subject):
        """Clean subject line for thread matching"""
        if not subject:
            return ""
        subject = re.sub(r'^(Re:|RE:|Fw:|FW:|Fwd:)\s*', '', subject, flags=re.IGNORECASE)
        subject = re.sub(r'\s+', ' ', subject).strip()
        return subject.lower()

    def group_by_threads(self):
        """Group emails into conversation threads"""
        threads = defaultdict(list)

        for email in self.emails:
            clean_subj = self.clean_subject(email.get('subject', ''))
            thread_key = clean_subj if clean_subj else f"no_subject_{email['email_id']}"
            threads[thread_key].append(email)

        for thread_key in threads:
            threads[thread_key].sort(key=lambda x: datetime.strptime(x['date'], '%d.%m.%Y %H:%M:%S'))

        self.threads = dict(threads)
        return self.threads

    def analyze_thread_patterns(self):
        """Analyze communication patterns within threads"""
        thread_stats = []

        for thread_key, emails in self.threads.items():
            if len(emails) < 2:
                continue

            participants = set()
            for email in emails:
                participants.add(email['from'])
                participants.add(email['to'])

            duration_days = 0
            if len(emails) > 1:
                start_date = datetime.strptime(emails[0]['date'], '%d.%m.%Y %H:%M:%S')
                end_date = datetime.strptime(emails[-1]['date'], '%d.%m.%Y %H:%M:%S')
                duration_days = (end_date - start_date).days

            thread_stats.append({
                'thread_key': thread_key,
                'email_count': len(emails),
                'participant_count': len(participants),
                'duration_days': duration_days,
                'participants': list(participants),
                'classifications': [email['classification'] for email in emails],
                'start_date': emails[0]['date'],
                'end_date': emails[-1]['date']
            })

        return sorted(thread_stats, key=lambda x: x['email_count'], reverse=True)


class TopicClusterer:
    """Perform topic clustering and classification analysis"""

    def __init__(self, email_data):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)

    def create_classification_dashboard(self):
        """Create topic dashboard based on existing classifications"""
        classification_stats = {}
        classifications = self.df['classification'].value_counts()

        for classification in classifications.index:
            class_emails = self.df[self.df['classification'] == classification]

            all_entities = defaultdict(list)
            for _, email in class_emails.iterrows():
                entities = email['entities']
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)

            top_entities = {}
            for entity_type, entity_list in all_entities.items():
                top_entities[entity_type] = Counter(entity_list).most_common(10)

            classification_stats[classification] = {
                'email_count': len(class_emails),
                'date_range': [class_emails['date'].min(), class_emails['date'].max()],
                'top_entities': top_entities,
                'sample_subjects': class_emails['subject'].head(5).tolist()
            }

        return classification_stats

    def semantic_clustering_simple(self, n_clusters=8):
        """Perform simple clustering using TF-IDF"""
        texts = []
        for email in self.emails:
            text = f"{email.get('subject', '')} {email.get('summary', '')}"
            texts.append(text)

        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append({'email': self.emails[i]})

            return self.analyze_clusters(clusters)
        except Exception as e:
            st.error(f"Clustering error: {e}")
            return {}

    def analyze_clusters(self, clusters):
        """Analyze and label clusters"""
        cluster_analysis = {}

        for cluster_id, cluster_emails in clusters.items():
            emails = [item['email'] for item in cluster_emails]

            all_words = []
            all_entities = defaultdict(list)
            classifications = []

            for email in emails:
                all_words.extend(email.get('summary', '').split())
                classifications.append(email['classification'])

                entities = email.get('entities', {})
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)

            word_freq = Counter(all_words)
            top_words = [word for word, count in word_freq.most_common(10)
                         if len(word) > 3 and word.isalpha()]

            top_entities = {}
            for entity_type, entity_list in all_entities.items():
                top_entities[entity_type] = Counter(entity_list).most_common(5)

            main_classification = Counter(classifications).most_common(1)[0][0]
            cluster_label = f"{main_classification}"
            if top_entities.get('topics'):
                cluster_label += f" - {top_entities['topics'][0][0]}"

            cluster_analysis[cluster_id] = {
                'label': cluster_label,
                'size': len(emails),
                'top_words': top_words[:5],
                'top_entities': top_entities,
                'main_classification': main_classification,
                'sample_emails': emails[:3],
                'date_range': [
                    min(email['date'] for email in emails),
                    max(email['date'] for email in emails)
                ]
            }

        return cluster_analysis


class StoryGenerator:
    """Generate keyword-based stories from email data"""

    def __init__(self, email_data):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self._prepare_search_corpus()

    def _prepare_search_corpus(self):
        """Prepare search corpus for similarity matching"""
        self.search_texts = []
        for email in self.emails:
            search_text = f"{email.get('subject', '')} {email.get('summary', '')}"
            entities = email.get('entities', {})
            for entity_type, entity_list in entities.items():
                search_text += " " + " ".join(entity_list)
            self.search_texts.append(search_text)

        if self.search_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)

    def find_relevant_emails(self, keyword: str, similarity_threshold: float = 0.1) -> List[tuple]:
        """Find emails relevant to a keyword with similarity scores"""
        if not hasattr(self, 'tfidf_matrix') or self.tfidf_matrix is None:
            return []

        keyword_vector = self.vectorizer.transform([keyword])
        similarities = cosine_similarity(keyword_vector, self.tfidf_matrix).flatten()

        relevant_emails = []
        for i, score in enumerate(similarities):
            if score > similarity_threshold:
                relevant_emails.append((i, score, self.emails[i]))

        # Also check for exact keyword matches
        keyword_lower = keyword.lower()
        for i, email in enumerate(self.emails):
            entities = email.get('entities', {})
            entity_match = any(keyword_lower in entity.lower()
                               for entity_list in entities.values()
                               for entity in entity_list)

            text_match = (keyword_lower in email.get('subject', '').lower() or
                          keyword_lower in email.get('summary', '').lower())

            if entity_match or text_match:
                existing_indices = [item[0] for item in relevant_emails]
                if i not in existing_indices:
                    boost_score = 0.5 if entity_match else 0.3
                    relevant_emails.append((i, boost_score, email))

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
                continue

        events.sort(key=lambda x: x.date)
        return events

    def extract_key_actors(self, events: List[StoryEvent]) -> tuple:
        """Extract key people and organizations from story events"""
        people_counter = Counter()
        org_counter = Counter()

        for event in events:
            people_counter[event.from_person] += 1
            people_counter[event.to_person] += 1

            entities = event.key_entities
            if 'people' in entities:
                for person in entities['people']:
                    people_counter[person] += 2

            if 'organizations' in entities:
                for org in entities['organizations']:
                    org_counter[org] += 1

        key_people = [person for person, count in people_counter.most_common(10) if person]
        key_orgs = [org for org, count in org_counter.most_common(5) if org]

        return key_people, key_orgs

    def generate_story_card(self, keyword: str, max_emails: int = 50) -> StoryCard:
        """Generate a complete story card for a given keyword"""
        relevant_emails = self.find_relevant_emails(keyword)

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

        timeline = self.create_story_timeline(relevant_emails)
        key_people, key_orgs = self.extract_key_actors(timeline)

        # Generate summary
        total_emails = len(timeline)
        duration = (timeline[-1].date - timeline[0].date).days if timeline else 0
        date_range_str = f"{timeline[0].date.strftime('%B %Y')} to {timeline[-1].date.strftime('%B %Y')}" if timeline else ""

        summary = f"The '{keyword}' story spans {duration} days from {date_range_str}, involving {total_emails} emails and {len(key_people)} key participants."
        if key_people:
            summary += f" Primary actors include: {', '.join(key_people[:3])}."
        if key_orgs:
            summary += f" Key organizations involved: {', '.join(key_orgs[:3])}."

        title = f"The {keyword.title()} Story"
        if key_orgs:
            title += f" - {key_orgs[0]}"

        date_range = (timeline[0].date, timeline[-1].date) if timeline else (None, None)
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


# Visualization Functions
def create_network_graph(story_card: StoryCard):
    """Create network graph of email communications for the story"""
    if not story_card.timeline:
        return None

    G = nx.Graph()
    edge_weights = defaultdict(int)

    for event in story_card.timeline:
        from_person = event.from_person.split('@')[0] if '@' in event.from_person else event.from_person
        to_person = event.to_person.split('@')[0] if '@' in event.to_person else event.to_person

        if from_person and to_person and from_person != to_person:
            edge_weights[(from_person, to_person)] += 1

    for (from_p, to_p), weight in edge_weights.items():
        G.add_edge(from_p, to_p, weight=weight)

    if len(G.nodes()) == 0:
        return None

    pos = nx.spring_layout(G, k=1, iterations=50)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_sizes = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_sizes.append(G.degree(node) * 10 + 20)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    ))

    fig.update_layout(
        title=f"Communication Network - {story_card.keyword.title()}",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


def create_timeline_chart(story_card: StoryCard):
    """Create timeline visualization"""
    if not story_card.timeline:
        return None

    dates = [event.date for event in story_card.timeline]
    subjects = [event.subject[:50] + "..." if len(event.subject) > 50 else event.subject for event in
                story_card.timeline]
    relevance_scores = [event.relevance_score for event in story_card.timeline]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=list(range(len(dates))),
        mode='markers+text',
        marker=dict(
            size=[score * 50 + 10 for score in relevance_scores],
            color=relevance_scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Relevance Score")
        ),
        text=subjects,
        textposition="middle right",
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Relevance: %{marker.color:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Timeline - {story_card.title}",
        xaxis_title="Date",
        yaxis_title="Email Sequence",
        height=400,
        showlegend=False
    )

    return fig


def create_topic_analysis(email_data):
    """Create topic analysis dashboard"""
    df = pd.DataFrame(email_data)

    classification_counts = df['classification'].value_counts()

    fig_pie = px.pie(
        values=classification_counts.values,
        names=classification_counts.index,
        title="Email Classifications Distribution",
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    df['date_parsed'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['date_parsed'])
    df['date_only'] = df['date_parsed'].dt.date

    daily_counts = df.groupby('date_only').size().reset_index(name='email_count')

    fig_timeline = px.line(
        daily_counts,
        x='date_only',
        y='email_count',
        title="Email Activity Over Time",
        color_discrete_sequence=['#667eea']
    )

    return fig_pie, fig_timeline


# Data Loading Functions
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    return [
        {
            "to": "john.doe@enron.com",
            "from": "jane.smith@enron.com",
            "date": "15.10.2001 14:30:00",
            "subject": "Dynegy merger discussions - confidential",
            "summary": "Discussion about potential merger with Dynegy including financial terms and regulatory considerations. The board is concerned about the valuation and timing.",
            "tone_analysis": "Professional",
            "classification": "Corporate Development",
            "entities": {
                "people": ["Kenneth Lay", "Jeffrey Skilling", "Andy Fastow"],
                "organizations": ["Dynegy Inc", "JP Morgan", "Chase Manhattan"],
                "locations": ["Houston", "New York", "Chicago"],
                "dates": ["15/10/2001"],
                "projects": ["Dynegy Merger"],
                "legal": ["SEC Filing", "Due Diligence"],
                "topics": ["merger", "acquisition", "valuation"]
            },
            "email_id": 1
        },
        {
            "to": "trading-desk@enron.com",
            "from": "tim.belden@enron.com",
            "date": "22.06.2000 09:15:00",
            "subject": "California power trading strategies",
            "summary": "Discussing new trading strategies for the California power market. Focus on peak hour pricing and congestion management.",
            "tone_analysis": "Business-focused",
            "classification": "Trading Operations",
            "entities": {
                "people": ["Tim Belden", "John Forney", "Jeff Richter"],
                "organizations": ["California ISO", "PG&E", "Southern California Edison"],
                "locations": ["California", "Los Angeles", "San Francisco"],
                "dates": ["22/06/2000"],
                "projects": ["California Trading"],
                "legal": ["FERC Regulations"],
                "topics": ["trading", "california", "power", "electricity"]
            },
            "email_id": 2
        },
        {
            "to": "legal-team@enron.com",
            "from": "vince.kaminski@enron.com",
            "date": "03.08.2001 16:45:00",
            "subject": "Risk management concerns - Raptor entities",
            "summary": "Expressing concerns about the risk profile of Raptor special purpose entities and their impact on financial statements.",
            "tone_analysis": "Concerned",
            "classification": "Risk Management",
            "entities": {
                "people": ["Vince Kaminski", "Rick Buy", "Greg Whalley"],
                "organizations": ["Arthur Andersen", "SEC", "Raptor"],
                "locations": ["Houston"],
                "dates": ["03/08/2001"],
                "projects": ["Raptor SPE"],
                "legal": ["SPE Structure", "Accounting Rules"],
                "topics": ["risk", "accounting", "entities", "compliance"]
            },
            "email_id": 3
        },
        {
            "to": "board-members@enron.com",
            "from": "kenneth.lay@enron.com",
            "date": "28.11.2001 11:20:00",
            "subject": "Merger termination - Dynegy deal collapsed",
            "summary": "Informing the board that Dynegy has terminated the merger agreement. Discussion of next steps and bankruptcy considerations.",
            "tone_analysis": "Urgent",
            "classification": "Crisis Management",
            "entities": {
                "people": ["Kenneth Lay", "Jeffrey Skilling", "Rebecca Mark"],
                "organizations": ["Dynegy Inc", "Bankruptcy Court", "Credit Rating Agencies"],
                "locations": ["Houston", "New York"],
                "dates": ["28/11/2001"],
                "projects": ["Merger Termination"],
                "legal": ["Bankruptcy Filing", "Merger Agreement"],
                "topics": ["bankruptcy", "merger", "crisis", "termination"]
            },
            "email_id": 4
        }
    ]


def load_data():
    """Load email data with various options"""
    if 'email_data' not in st.session_state:
        st.session_state.email_data = None

    st.sidebar.markdown("### 📁 Data Loading")

    # Option 1: File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload JSON file",
        type=['json'],
        help="Upload your Enron emails JSON file"
    )

    if uploaded_file is not None:
        try:
            st.session_state.email_data = json.load(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(st.session_state.email_data)} emails!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {e}")
            st.session_state.email_data = None

    # Option 2: Use sample data
    if st.sidebar.button("🎭 Use Sample Data"):
        st.session_state.email_data = load_sample_data()
        st.sidebar.success("✅ Sample data loaded!")

    # Option 3: Manual JSON input
    with st.sidebar.expander("📝 Paste JSON Data"):
        json_text = st.text_area("Paste your JSON here:", height=100)
        if st.button("Load JSON Text") and json_text:
            try:
                st.session_state.email_data = json.loads(json_text)
                st.sidebar.success(f"✅ Loaded {len(st.session_state.email_data)} emails!")
            except Exception as e:
                st.sidebar.error(f"❌ Invalid JSON: {e}")

    return st.session_state.email_data


# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📧 Enron Story Explorer</h1>
        <p><strong>Discover Hidden Stories • Analyze Communication Patterns • Explore Email Networks</strong></p>
        <p>Complete toolkit for analyzing the Enron email dataset with AI-powered story generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    email_data = load_data()

    if not email_data:
        st.info("👆 **Get Started:** Use the sidebar to load your data or try the sample dataset!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 📤 Upload Data
            - JSON format required
            - Supports large datasets
            - Secure local processing
            """)

        with col2:
            st.markdown("""
            ### 🎭 Try Sample Data
            - Pre-loaded Enron emails
            - Perfect for testing
            - Includes all features
            """)

        with col3:
            st.markdown("""
            ### 🔍 Key Features
            - Story generation
            - Network analysis  
            - Topic clustering
            - Interactive visualizations
            """)
        return

    # Initialize analyzers
    if 'analyzers_initialized' not in st.session_state:
        with st.spinner("🧠 Initializing AI analyzers..."):
            st.session_state.thread_analyzer = EmailThreadAnalyzer(email_data)
            st.session_state.topic_clusterer = TopicClusterer(email_data)
            st.session_state.story_generator = StoryGenerator(email_data)
            st.session_state.analyzers_initialized = True

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Story Search")

    suggested_keywords = [
        "merger", "california", "trading", "dynegy",
        "crisis", "pipeline", "SEC", "investigation",
        "accounting", "bankruptcy", "power", "risk"
    ]

    selected_suggestion = st.sidebar.selectbox(
        "Quick Search:",
        [""] + suggested_keywords,
        format_func=lambda x: "Select a keyword..." if x == "" else x.title()
    )

    custom_keyword = st.sidebar.text_input("Or enter custom keyword:")
    search_keyword = custom_keyword if custom_keyword else selected_suggestion

    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_emails = st.slider("Max emails per story", 10, 100, 50)
        similarity_threshold = st.slider("Similarity threshold", 0.05, 0.5, 0.1)
        n_clusters = st.slider("Number of topic clusters", 3, 15, 8)

    # Search button
    if st.sidebar.button("🚀 Generate Story", type="primary") and search_keyword:
        with st.spinner(f"🔍 Analyzing emails for '{search_keyword}'..."):
            story_card = st.session_state.story_generator.generate_story_card(
                search_keyword, max_emails=max_emails
            )
            st.session_state.current_story = story_card

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Story Explorer",
        "🧵 Thread Analysis",
        "🏷️ Topic Clusters",
        "🌐 Network Analysis",
        "📊 Analytics Dashboard"
    ])

    with tab1:
        st.markdown("### 📖 Interactive Story Explorer")

        if hasattr(st.session_state, 'current_story'):
            story = st.session_state.current_story

            # Story header
            st.markdown(f"## {story.title}")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📧 Total Emails", story.total_emails)
            with col2:
                st.metric("👥 Key People", len(story.key_people))
            with col3:
                st.metric("🏢 Organizations", len(story.key_organizations))
            with col4:
                if story.date_range[0]:
                    duration = (story.date_range[1] - story.date_range[0]).days
                    st.metric("📅 Duration (days)", duration)

            # Story summary
            st.markdown(f"""
            <div class="story-card">
                <h3>📋 Story Summary</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">{story.summary}</p>
            </div>
            """, unsafe_allow_html=True)

            # Key actors
            if story.key_people or story.key_organizations:
                col1, col2 = st.columns(2)

                with col1:
                    if story.key_people:
                        st.markdown("#### 👥 Key People")
                        for person in story.key_people[:10]:
                            st.markdown(f"• **{person}**")

                with col2:
                    if story.key_organizations:
                        st.markdown("#### 🏢 Key Organizations")
                        for org in story.key_organizations[:5]:
                            st.markdown(f"• **{org}**")

            # Timeline visualization
            if story.timeline:
                st.markdown("### 📅 Story Timeline")

                timeline_fig = create_timeline_chart(story)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)

                # Detailed timeline
                st.markdown("### 📝 Email Timeline Details")

                for i, event in enumerate(story.timeline[:20]):
                    with st.expander(f"📧 {event.date.strftime('%Y-%m-%d')} - {event.subject}"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**From:** {event.from_person}")
                            st.markdown(f"**To:** {event.to_person}")
                            st.markdown(f"**Summary:** {event.summary}")

                            # Show entities
                            if event.key_entities:
                                st.markdown("**Key Entities:**")
                                for entity_type, entities in event.key_entities.items():
                                    if entities:
                                        entity_tags = "".join(
                                            [f'<span class="keyword-tag">{entity}</span>' for entity in entities[:3]])
                                        st.markdown(f"*{entity_type.title()}:* {entity_tags}", unsafe_allow_html=True)

                        with col2:
                            st.metric("Relevance Score", f"{event.relevance_score:.3f}")
                            st.markdown(f"**Classification:** {event.email_data.get('classification', 'N/A')}")

        else:
            st.info("👆 **Select a keyword from the sidebar to start exploring stories!**")

            st.markdown("### 🎯 Try These Popular Keywords:")

            cols = st.columns(4)
            for i, keyword in enumerate(suggested_keywords[:8]):
                with cols[i % 4]:
                    if st.button(f"🔍 **{keyword.title()}**", key=f"btn_{keyword}"):
                        with st.spinner(f"Analyzing '{keyword}'..."):
                            story_card = st.session_state.story_generator.generate_story_card(keyword)
                            st.session_state.current_story = story_card
                            st.rerun()

    with tab2:
        st.markdown("### 🧵 Email Thread Analysis")

        with st.spinner("Analyzing conversation threads..."):
            threads = st.session_state.thread_analyzer.group_by_threads()
            thread_stats = st.session_state.thread_analyzer.analyze_thread_patterns()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Threads", len(threads))
        with col2:
            avg_thread_length = np.mean([len(emails) for emails in threads.values()])
            st.metric("Avg Thread Length", f"{avg_thread_length:.1f}")
        with col3:
            long_threads = len([t for t in thread_stats if t['email_count'] > 5])
            st.metric("Long Threads (5+ emails)", long_threads)

        # Top threads
        st.markdown("#### 📈 Longest Conversation Threads")

        for i, thread in enumerate(thread_stats[:10]):
            st.markdown(f"""
            <div class="timeline-item">
                <h4>{i + 1}. {thread['thread_key'][:80]}{'...' if len(thread['thread_key']) > 80 else ''}</h4>
                <p><strong>📧 {thread['email_count']} emails</strong> • 
                   <strong>👥 {thread['participant_count']} participants</strong> • 
                   <strong>📅 {thread['duration_days']} days</strong></p>
                <p><strong>Participants:</strong> {', '.join(thread['participants'][:5])}{'...' if len(thread['participants']) > 5 else ''}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🏷️ Topic Clustering Analysis")

        # Classification dashboard
        with st.spinner("Analyzing topic classifications..."):
            classification_stats = st.session_state.topic_clusterer.create_classification_dashboard()

        st.markdown("#### 📊 Email Classifications")

        for classification, stats in classification_stats.items():
            st.markdown(f"""
            <div class="cluster-card">
                <h4>{classification}</h4>
                <p><strong>📧 {stats['email_count']} emails</strong></p>
                <p><strong>📅 Date Range:</strong> {stats['date_range'][0]} to {stats['date_range'][1]}</p>
                <details>
                    <summary><strong>Sample Subjects</strong></summary>
                    <ul>
                        {''.join([f'<li>{subject}</li>' for subject in stats['sample_subjects']])}
                    </ul>
                </details>
            </div>
            """, unsafe_allow_html=True)

        # Semantic clustering
        st.markdown("#### 🔬 Semantic Clustering")

        with st.spinner("Performing semantic clustering..."):
            cluster_analysis = st.session_state.topic_clusterer.semantic_clustering_simple(n_clusters=n_clusters)

        if cluster_analysis:
            for cluster_id, analysis in cluster_analysis.items():
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>Cluster {cluster_id + 1}: {analysis['label']}</h4>
                    <p><strong>📧 {analysis['size']} emails</strong></p>
                    <p><strong>🎯 Main Classification:</strong> {analysis['main_classification']}</p>
                    <p><strong>🔑 Key Terms:</strong> {', '.join(analysis['top_words'])}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 🌐 Communication Network Analysis")

        if hasattr(st.session_state, 'current_story'):
            story = st.session_state.current_story

            if story.timeline:
                st.markdown(f"#### Network for '{story.keyword.title()}' Story")

                network_fig = create_network_graph(story)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)

                    # Network statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        unique_senders = len(set(event.from_person for event in story.timeline))
                        st.metric("Unique Senders", unique_senders)
                    with col2:
                        unique_recipients = len(set(event.to_person for event in story.timeline))
                        st.metric("Unique Recipients", unique_recipients)
                    with col3:
                        total_connections = len(story.timeline)
                        st.metric("Total Communications", total_connections)
                else:
                    st.info("Not enough communication data to create network graph.")
            else:
                st.info("No timeline data available for network analysis.")
        else:
            st.info("🔍 **Select a story to view its communication network.**")

            # Show overall network stats
            st.markdown("#### 📊 Overall Dataset Network Statistics")

            df = pd.DataFrame(email_data)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Unique Senders", df['from'].nunique())
            with col2:
                st.metric("Unique Recipients", df['to'].nunique())
            with col3:
                st.metric("Total Emails", len(df))
            with col4:
                unique_pairs = len(set(zip(df['from'], df['to'])))
                st.metric("Unique Connections", unique_pairs)

    with tab5:
        st.markdown("### 📊 Analytics Dashboard")

        # Topic analysis charts
        fig_pie, fig_timeline = create_topic_analysis(email_data)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Dataset statistics
        st.markdown("#### 📈 Dataset Overview")

        df = pd.DataFrame(email_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📧 Total Emails", len(df))
        with col2:
            st.metric("👥 Unique Senders", df['from'].nunique())
        with col3:
            try:
                date_range = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                duration = (date_range.max() - date_range.min()).days
                st.metric("📅 Date Range (days)", duration)
            except:
                st.metric("📅 Date Range", "N/A")
        with col4:
            st.metric("🏷️ Classifications", df['classification'].nunique())

        # Top entities analysis
        st.markdown("#### 🎯 Most Mentioned Entities")

        all_entities = defaultdict(Counter)
        for email in email_data:
            entities = email.get('entities', {})
            for entity_type, entity_list in entities.items():
                all_entities[entity_type].update(entity_list)

        entity_tabs = st.tabs(["👥 People", "🏢 Organizations", "📍 Locations", "🔑 Topics"])

        with entity_tabs[0]:
            if 'people' in all_entities:
                people_df = pd.DataFrame(all_entities['people'].most_common(10), columns=['Person', 'Mentions'])
                fig = px.bar(people_df, x='Mentions', y='Person', orientation='h', title="Most Mentioned People")
                st.plotly_chart(fig, use_container_width=True)

        with entity_tabs[1]:
            if 'organizations' in all_entities:
                orgs_df = pd.DataFrame(all_entities['organizations'].most_common(10),
                                       columns=['Organization', 'Mentions'])
                fig = px.bar(orgs_df, x='Mentions', y='Organization', orientation='h',
                             title="Most Mentioned Organizations")
                st.plotly_chart(fig, use_container_width=True)

        with entity_tabs[2]:
            if 'locations' in all_entities:
                locs_df = pd.DataFrame(all_entities['locations'].most_common(10), columns=['Location', 'Mentions'])
                fig = px.bar(locs_df, x='Mentions', y='Location', orientation='h', title="Most Mentioned Locations")
                st.plotly_chart(fig, use_container_width=True)

        with entity_tabs[3]:
            if 'topics' in all_entities:
                topics_df = pd.DataFrame(all_entities['topics'].most_common(10), columns=['Topic', 'Mentions'])
                fig = px.bar(topics_df, x='Mentions', y='Topic', orientation='h', title="Most Mentioned Topics")
                st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>Enron Story Explorer</strong> | AI-Powered Email Analysis | 
        Built with Streamlit, Plotly, and scikit-learn</p>
        <p>📊 Discover hidden narratives in complex email datasets</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
