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
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .story-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    .timeline-item {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


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


class StreamlitStoryGenerator:
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

        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)

    def find_relevant_emails(self, keyword: str, similarity_threshold: float = 0.1) -> List[tuple]:
        """Find emails relevant to a keyword with similarity scores"""
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


def load_data():
    """Load email data - replace with your data loading logic"""
    if 'email_data' not in st.session_state:
        # For demo purposes, create sample data
        # Replace this with: st.session_state.email_data = json.load(open('your_file.json'))
        sample_data = [
            {
                "to": "john.doe@enron.com",
                "from": "jane.smith@enron.com",
                "date": "15.10.2001 14:30:00",
                "subject": "Dynegy merger discussions - confidential",
                "summary": "Discussion about potential merger with Dynegy including financial terms and regulatory considerations.",
                "tone_analysis": "Professional",
                "classification": "Corporate Development",
                "entities": {
                    "people": ["Kenneth Lay", "Jeffrey Skilling"],
                    "organizations": ["Dynegy Inc", "JP Morgan"],
                    "locations": ["Houston", "New York"],
                    "dates": ["15/10/2001"],
                    "projects": ["Dynegy Merger"],
                    "legal": ["SEC Filing"],
                    "topics": ["merger", "acquisition"]
                },
                "email_id": 1
            },
            # Add more sample data here...
        ]

        # Try to load real data, fall back to sample
        try:
            uploaded_file = st.file_uploader("Upload your Enron emails JSON file", type=['json'])
            if uploaded_file is not None:
                st.session_state.email_data = json.load(uploaded_file)
                st.success(f"Loaded {len(st.session_state.email_data)} emails!")
            else:
                st.session_state.email_data = sample_data
                st.info("Using sample data. Upload your JSON file to use real data.")
        except Exception as e:
            st.session_state.email_data = sample_data
            st.warning(f"Could not load data: {e}. Using sample data.")

    return st.session_state.email_data


def create_network_graph(story_card: StoryCard):
    """Create network graph of email communications for the story"""
    if not story_card.timeline:
        return None

    G = nx.Graph()
    edge_weights = defaultdict(int)

    # Add edges for each email
    for event in story_card.timeline:
        from_person = event.from_person.split('@')[0] if '@' in event.from_person else event.from_person
        to_person = event.to_person.split('@')[0] if '@' in event.to_person else event.to_person

        if from_person and to_person and from_person != to_person:
            edge_weights[(from_person, to_person)] += 1

    # Add edges to graph
    for (from_p, to_p), weight in edge_weights.items():
        G.add_edge(from_p, to_p, weight=weight)

    if len(G.nodes()) == 0:
        return None

    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create edges
    edge_x = []
    edge_y = []
    edge_weights_list = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights_list.append(G[edge[0]][edge[1]]['weight'])

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Size based on degree centrality
        node_sizes.append(G.degree(node) * 10 + 20)

    # Create plotly figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
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
        annotations=[dict(
            text="Network shows email communication patterns for this story",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


def create_timeline_chart(story_card: StoryCard):
    """Create timeline visualization"""
    if not story_card.timeline:
        return None

    # Prepare data
    dates = [event.date for event in story_card.timeline]
    subjects = [event.subject[:50] + "..." if len(event.subject) > 50 else event.subject for event in
                story_card.timeline]
    relevance_scores = [event.relevance_score for event in story_card.timeline]

    # Create timeline chart
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

    # Classification distribution
    classification_counts = df['classification'].value_counts()

    fig_pie = px.pie(
        values=classification_counts.values,
        names=classification_counts.index,
        title="Email Classifications Distribution"
    )

    # Timeline of email activity
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
    df['date_only'] = df['date_parsed'].dt.date

    daily_counts = df.groupby('date_only').size().reset_index(name='email_count')

    fig_timeline = px.line(
        daily_counts,
        x='date_only',
        y='email_count',
        title="Email Activity Over Time"
    )

    return fig_pie, fig_timeline


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📧 Enron Story Explorer</h1>
        <p>Discover hidden stories within the Enron email dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    email_data = load_data()

    if not email_data:
        st.error("No data loaded. Please upload your JSON file.")
        return

    # Initialize story generator
    if 'story_generator' not in st.session_state:
        st.session_state.story_generator = StreamlitStoryGenerator(email_data)

    # Sidebar
    st.sidebar.title("🔍 Story Search")

    # Suggested keywords
    suggested_keywords = ["merger", "california", "trading", "dynegy", "crisis", "pipeline", "SEC", "investigation"]

    selected_suggestion = st.sidebar.selectbox(
        "Quick Search:",
        [""] + suggested_keywords,
        format_func=lambda x: "Select a keyword..." if x == "" else x.title()
    )

    # Custom keyword input
    custom_keyword = st.sidebar.text_input("Or enter custom keyword:")

    # Use selected or custom keyword
    search_keyword = custom_keyword if custom_keyword else selected_suggestion

    # Search button
    if st.sidebar.button("🚀 Generate Story") and search_keyword:
        with st.spinner(f"Analyzing emails for '{search_keyword}'..."):
            story_card = st.session_state.story_generator.generate_story_card(search_keyword)
            st.session_state.current_story = story_card

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Story Explorer", "📊 Analytics", "🌐 Network", "📈 Dashboard"])

    with tab1:
        if hasattr(st.session_state, 'current_story'):
            story = st.session_state.current_story

            # Story header
            st.title(story.title)

            # Metrics
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
                <p>{story.summary}</p>
            </div>
            """, unsafe_allow_html=True)

            # Key actors
            if story.key_people or story.key_organizations:
                col1, col2 = st.columns(2)

                with col1:
                    if story.key_people:
                        st.subheader("👥 Key People")
                        for person in story.key_people[:10]:
                            st.write(f"• {person}")

                with col2:
                    if story.key_organizations:
                        st.subheader("🏢 Key Organizations")
                        for org in story.key_organizations[:5]:
                            st.write(f"• {org}")

            # Timeline
            if story.timeline:
                st.subheader("📅 Email Timeline")

                # Show timeline chart
                timeline_fig = create_timeline_chart(story)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)

                # Detailed timeline
                st.subheader("📝 Detailed Timeline")
                for i, event in enumerate(story.timeline[:20]):  # Show first 20 events
                    with st.expander(f"{event.date.strftime('%Y-%m-%d')} - {event.subject}"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**From:** {event.from_person}")
                            st.write(f"**To:** {event.to_person}")
                            st.write(f"**Summary:** {event.summary}")
                        with col2:
                            st.metric("Relevance Score", f"{event.relevance_score:.2f}")

                            # Show entities if available
                            if event.key_entities:
                                for entity_type, entities in event.key_entities.items():
                                    if entities:
                                        st.write(f"**{entity_type.title()}:** {', '.join(entities[:3])}")

        else:
            st.info("👆 Select a keyword from the sidebar to start exploring stories!")

            # Show some sample stories
            st.subheader("🎯 Try These Popular Keywords:")

            cols = st.columns(3)
            for i, keyword in enumerate(suggested_keywords[:6]):
                with cols[i % 3]:
                    if st.button(f"🔍 {keyword.title()}", key=f"btn_{keyword}"):
                        with st.spinner(f"Analyzing '{keyword}'..."):
                            story_card = st.session_state.story_generator.generate_story_card(keyword)
                            st.session_state.current_story = story_card
                            st.rerun()

    with tab2:
        st.subheader("📊 Email Analytics")

        # Create topic analysis
        fig_pie, fig_timeline = create_topic_analysis(email_data)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Data statistics
        df = pd.DataFrame(email_data)
        st.subheader("📈 Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", len(df))
        with col2:
            unique_senders = df['from'].nunique()
            st.metric("Unique Senders", unique_senders)
        with col3:
            date_range = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
            duration = (date_range.max() - date_range.min()).days
            st.metric("Date Range (days)", duration)
        with col4:
            classifications = df['classification'].nunique()
            st.metric("Classifications", classifications)

    with tab3:
        st.subheader("🌐 Communication Networks")

        if hasattr(st.session_state, 'current_story'):
            story = st.session_state.current_story

            if story.timeline:
                # Create network graph
                network_fig = create_network_graph(story)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.info("Not enough communication data to create network graph.")
            else:
                st.info("Select a story to view its communication network.")
        else:
            st.info("Select a keyword to explore communication networks.")

    with tab4:
        st.subheader("📈 Story Dashboard")

        # Generate stories for multiple keywords
        dashboard_keywords = ["merger", "california", "trading", "crisis"]

        stories_data = []
        for keyword in dashboard_keywords:
            story = st.session_state.story_generator.generate_story_card(keyword)
            if story.total_emails > 0:
                stories_data.append({
                    'Keyword': keyword.title(),
                    'Total Emails': story.total_emails,
                    'Key People': len(story.key_people),
                    'Organizations': len(story.key_organizations),
                    'Duration (days)': (story.date_range[1] - story.date_range[0]).days if story.date_range[0] else 0
                })

        if stories_data:
            df_stories = pd.DataFrame(stories_data)

            # Stories comparison chart
            fig_comparison = px.bar(
                df_stories,
                x='Keyword',
                y='Total Emails',
                title="Story Sizes Comparison",
                color='Total Emails',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Stories table
            st.subheader("📊 Stories Summary")
            st.dataframe(df_stories, use_container_width=True)

        # Overall statistics
        st.subheader("🎯 Quick Insights")
        total_emails = len(email_data)
        df = pd.DataFrame(email_data)

        insights = [
            f"📧 **{total_emails:,}** total emails in the dataset",
            f"👥 **{df['from'].nunique():,}** unique email senders",
            f"🏷️ **{df['classification'].nunique()}** different email classifications",
            f"📅 Dataset spans from **{df['date'].min()}** to **{df['date'].max()}**"
        ]

        for insight in insights:
            st.markdown(insight)


if __name__ == "__main__":
    main()
