import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict
import networkx as nx

# Set page config
st.set_page_config(
    page_title="Efficient Enron Email Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .story-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)


class EfficientEnronAnalyzer:
    def __init__(self):
        self.emails = None
        self.stories = None
        self.entity_graph = None

    def load_data(self, uploaded_file):
        """Load email data from JSON file"""
        try:
            if uploaded_file is not None:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    self.emails = pd.DataFrame(data)
                else:
                    self.emails = pd.DataFrame([data])

                # Convert date strings to datetime
                self.emails['date'] = pd.to_datetime(self.emails['date'], format='%d.%m.%Y %H:%M:%S')

                # Extract entities efficiently
                self._extract_all_entities()
                return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def _extract_all_entities(self):
        """Extract and flatten all entities for efficient access"""
        self.emails['people'] = self.emails['entities'].apply(
            lambda x: x.get('people', []) if isinstance(x, dict) else []
        )
        self.emails['organizations'] = self.emails['entities'].apply(
            lambda x: x.get('organizations', []) if isinstance(x, dict) else []
        )
        self.emails['topics'] = self.emails['entities'].apply(
            lambda x: x.get('topics', []) if isinstance(x, dict) else []
        )
        self.emails['projects'] = self.emails['entities'].apply(
            lambda x: x.get('projects', []) if isinstance(x, dict) else []
        )
        self.emails['locations'] = self.emails['entities'].apply(
            lambda x: x.get('locations', []) if isinstance(x, dict) else []
        )

    def generate_stories_efficiently(self):
        """Generate stories using entity-based clustering and temporal patterns"""
        stories = []

        # 1. Project-based stories
        project_stories = self._find_project_stories()
        stories.extend(project_stories)

        # 2. Topic-based temporal clusters
        topic_stories = self._find_topic_stories()
        stories.extend(topic_stories)

        # 3. Key participant stories
        participant_stories = self._find_participant_stories()
        stories.extend(participant_stories)

        # 4. Time-based burst detection
        burst_stories = self._find_burst_stories()
        stories.extend(burst_stories)

        # Sort stories by importance score
        stories.sort(key=lambda x: x['importance_score'], reverse=True)
        self.stories = stories[:10]  # Top 10 stories

        return self.stories

    def _find_project_stories(self):
        """Find stories based on projects mentioned"""
        stories = []

        # Get all emails with projects
        project_emails = self.emails[self.emails['projects'].apply(len) > 0].copy()

        if len(project_emails) == 0:
            return stories

        # Group by project
        project_groups = defaultdict(list)
        for idx, email in project_emails.iterrows():
            for project in email['projects']:
                project_groups[project].append(idx)

        # Create stories for significant projects
        for project, email_indices in project_groups.items():
            if len(email_indices) >= 2:  # At least 2 emails
                project_emails_subset = self.emails.iloc[email_indices]

                story = {
                    'type': 'project',
                    'title': f"Project: {project}",
                    'email_count': len(email_indices),
                    'participants': self._get_unique_participants(project_emails_subset),
                    'organizations': self._get_all_organizations(project_emails_subset),
                    'date_range': (project_emails_subset['date'].min(), project_emails_subset['date'].max()),
                    'duration_days': (project_emails_subset['date'].max() - project_emails_subset['date'].min()).days,
                    'summary': self._generate_story_summary(project_emails_subset, context=f"project '{project}'"),
                    'timeline': project_emails_subset[['date', 'from', 'to', 'subject', 'summary']].to_dict('records'),
                    'importance_score': len(email_indices) * 2 + len(
                        self._get_unique_participants(project_emails_subset))
                }
                stories.append(story)

        return stories

    def _find_topic_stories(self):
        """Find stories based on topics with temporal clustering"""
        stories = []

        # Get topic frequencies
        all_topics = []
        for topics in self.emails['topics']:
            all_topics.extend(topics)

        topic_counter = Counter(all_topics)

        # Analyze significant topics
        for topic, count in topic_counter.most_common(10):
            if count >= 3:  # At least 3 mentions
                topic_emails = self.emails[self.emails['topics'].apply(lambda x: topic in x)]

                # Check for temporal clustering
                dates = pd.to_datetime(topic_emails['date'])
                date_diffs = dates.diff().dt.days.dropna()

                # If emails are clustered in time (avg gap < 7 days)
                if len(date_diffs) > 0 and date_diffs.mean() < 7:
                    story = {
                        'type': 'topic_cluster',
                        'title': f"Topic Surge: {topic}",
                        'email_count': len(topic_emails),
                        'participants': self._get_unique_participants(topic_emails),
                        'organizations': self._get_all_organizations(topic_emails),
                        'date_range': (topic_emails['date'].min(), topic_emails['date'].max()),
                        'duration_days': (topic_emails['date'].max() - topic_emails['date'].min()).days,
                        'summary': self._generate_story_summary(topic_emails, context=f"topic '{topic}'"),
                        'timeline': topic_emails[['date', 'from', 'to', 'subject', 'summary']].head(5).to_dict(
                            'records'),
                        'importance_score': count * 1.5 + (10 - date_diffs.mean() if len(date_diffs) > 0 else 0)
                    }
                    stories.append(story)

        return stories

    def _find_participant_stories(self):
        """Find stories based on key participants"""
        stories = []

        # Get most active senders
        sender_counts = self.emails['from'].value_counts()

        for sender, count in sender_counts.head(5).items():
            if count >= 5:  # At least 5 emails
                sender_emails = self.emails[self.emails['from'] == sender]

                # Analyze their communication patterns
                recipients = []
                for to in sender_emails['to']:
                    if isinstance(to, str):
                        recipients.append(to)
                    elif isinstance(to, list):
                        recipients.extend(to)

                unique_recipients = list(set(recipients))

                story = {
                    'type': 'key_participant',
                    'title': f"Key Player: {sender.split('@')[0]}",
                    'email_count': count,
                    'participants': unique_recipients[:10],  # Top 10 recipients
                    'organizations': self._get_all_organizations(sender_emails),
                    'date_range': (sender_emails['date'].min(), sender_emails['date'].max()),
                    'duration_days': (sender_emails['date'].max() - sender_emails['date'].min()).days,
                    'summary': f"{sender.split('@')[0]} sent {count} emails to {len(unique_recipients)} different people, primarily discussing: {', '.join(self._get_top_topics(sender_emails, 3))}",
                    'timeline': sender_emails[['date', 'to', 'subject', 'tone_analysis']].head(5).to_dict('records'),
                    'importance_score': count + len(unique_recipients) * 0.5
                }
                stories.append(story)

        return stories

    def _find_burst_stories(self):
        """Find stories based on email bursts"""
        stories = []

        # Group by date
        daily_counts = self.emails.groupby(self.emails['date'].dt.date).size()

        # Find days with unusually high activity
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        threshold = mean_count + 2 * std_count

        burst_dates = daily_counts[daily_counts > threshold].index

        for burst_date in burst_dates:
            # Get emails from this date and surrounding days
            start_date = burst_date - timedelta(days=1)
            end_date = burst_date + timedelta(days=1)

            burst_emails = self.emails[
                (self.emails['date'].dt.date >= start_date) &
                (self.emails['date'].dt.date <= end_date)
                ]

            if len(burst_emails) >= 5:
                story = {
                    'type': 'activity_burst',
                    'title': f"Activity Spike: {burst_date.strftime('%Y-%m-%d')}",
                    'email_count': len(burst_emails),
                    'participants': self._get_unique_participants(burst_emails),
                    'organizations': self._get_all_organizations(burst_emails),
                    'date_range': (burst_emails['date'].min(), burst_emails['date'].max()),
                    'duration_days': 3,
                    'summary': self._generate_story_summary(burst_emails, context="email burst"),
                    'timeline': burst_emails[['date', 'from', 'to', 'subject']].to_dict('records'),
                    'importance_score': len(burst_emails) * 1.2
                }
                stories.append(story)

        return stories

    def _get_unique_participants(self, email_subset):
        """Get unique participants from email subset"""
        participants = set()
        participants.update(email_subset['from'].unique())

        for to_list in email_subset['to']:
            if isinstance(to_list, str):
                participants.add(to_list)
            elif isinstance(to_list, list):
                participants.update(to_list)

        return list(participants)

    def _get_all_organizations(self, email_subset):
        """Get all organizations mentioned"""
        orgs = []
        for org_list in email_subset['organizations']:
            orgs.extend(org_list)
        return list(set(orgs))

    def _get_top_topics(self, email_subset, n=3):
        """Get top N topics from email subset"""
        all_topics = []
        for topic_list in email_subset['topics']:
            all_topics.extend(topic_list)

        if not all_topics:
            return ["general communication"]

        topic_counts = Counter(all_topics)
        return [topic for topic, _ in topic_counts.most_common(n)]

    def _generate_project_summary(self, project_name, email_subset):
        """Generate summary for a project"""
        participants = self._get_unique_participants(email_subset)
        orgs = self._get_all_organizations(email_subset)
        topics = self._get_top_topics(email_subset)

        duration = (email_subset['date'].max() - email_subset['date'].min()).days

        summary = f"{project_name} involved {len(participants)} people"
        if orgs:
            summary += f" from organizations including {', '.join(orgs[:2])}"
        summary += f" over {duration} days. "
        if topics:
            summary += f"Key topics discussed: {', '.join(topics)}."

        return summary

    def create_entity_network(self):
        """Create efficient entity-based network"""
        G = nx.Graph()

        # Add nodes and edges based on shared entities
        for idx, email in self.emails.iterrows():
            sender = email['from']

            # Connect sender to recipients
            recipients = [email['to']] if isinstance(email['to'], str) else email['to']
            for recipient in recipients:
                if pd.notna(recipient):
                    G.add_edge(sender, recipient, weight=G.get_edge_data(sender, recipient, {}).get('weight', 0) + 1)

            # Connect people through shared projects
            for project in email['projects']:
                G.add_node(project, node_type='project')
                G.add_edge(sender, project, edge_type='involved_in')

            # Connect people through shared organizations
            for org in email['organizations']:
                G.add_node(org, node_type='organization')
                G.add_edge(sender, org, edge_type='associated_with')

        self.entity_graph = G
        return G

    def visualize_timeline(self):
        """Create efficient timeline visualization"""
        daily_counts = self.emails.groupby(self.emails['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']

        fig = px.line(daily_counts, x='date', y='count',
                      title='Email Activity Timeline',
                      labels={'count': 'Number of Emails', 'date': 'Date'})

        fig.update_layout(
            hovermode='x unified',
            height=400
        )

        return fig

    def visualize_entity_relationships(self):
        """Visualize entity relationships efficiently"""
        if self.entity_graph is None:
            self.create_entity_network()

        G = self.entity_graph

        # Filter to most important nodes
        degree_dict = dict(G.degree())
        important_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:50]
        important_node_names = [node for node, degree in important_nodes]

        # Create subgraph
        subgraph = G.subgraph(important_node_names)

        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)

        # Create traces
        edge_x = []
        edge_y = []

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Node trace
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []

        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Color by type
            node_type = subgraph.nodes[node].get('node_type', 'person')
            if node_type == 'project':
                node_color.append('#ff7f0e')
            elif node_type == 'organization':
                node_color.append('#2ca02c')
            else:
                node_color.append('#1f77b4')

            # Size by degree
            node_size.append(min(subgraph.degree(node) * 3 + 10, 30))
            node_text.append(node.split('@')[0] if '@' in node else node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=node_text
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Key Entity Relationships",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            annotations=[
                dict(
                    text="🔵 People | 🟠 Projects | 🟢 Organizations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    xanchor='center'
                )
            ]
        )

        return fig

    def _generate_story_summary(self, email_subset, context="project"):
        """Generate a readable story-style summary from emails in the subset"""
        if email_subset.empty:
            return "No meaningful summary could be generated due to lack of data."

        participants = self._get_unique_participants(email_subset)
        organizations = self._get_all_organizations(email_subset)
        topics = self._get_top_topics(email_subset)

        emails_sorted = email_subset.sort_values(by="date").head(5)
        messages = []

        for _, row in emails_sorted.iterrows():
            sender = row['from'].split('@')[0] if '@' in row['from'] else row['from']
            subject = row['subject'] if isinstance(row['subject'], str) else ""
            summary = row['summary'] if isinstance(row['summary'], str) else ""
            date_str = row['date'].strftime('%b %d, %Y')
            messages.append(f"On {date_str}, {sender} wrote: \"{summary}\"")

        topic_str = f"The main themes included: {', '.join(topics)}." if topics else ""
        org_str = f"Key organizations involved were: {', '.join(organizations[:2])}." if organizations else ""

        final_summary = (
                f"During this {context}-related communication, a group of {len(participants)} individuals "
                f"corresponded over a span of {(email_subset['date'].max() - email_subset['date'].min()).days} days. "
                f"{org_str} {topic_str} The discussion evolved as follows:\n\n" +
                "\n".join(messages)
        )

        return final_summary


def main():
    st.markdown('<h1 class="main-header">📧 Efficient Enron Email Analysis</h1>', unsafe_allow_html=True)

    analyzer = EfficientEnronAnalyzer()

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload JSON email data", type=['json'])

        if uploaded_file:
            if analyzer.load_data(uploaded_file):
                st.success(f"✅ Loaded {len(analyzer.emails)} emails")

                if st.button("🚀 Generate Analysis", type="primary"):
                    with st.spinner("Analyzing emails..."):
                        stories = analyzer.generate_stories_efficiently()
                    st.success("✨ Analysis complete!")

    if analyzer.emails is not None:
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📚 Stories", "🕸️ Network"])

        with tab1:
            st.header("Dataset Overview")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Emails", len(analyzer.emails))
            with col2:
                st.metric("Unique Senders", analyzer.emails['from'].nunique())
            with col3:
                st.metric("Date Range (Days)",
                          (analyzer.emails['date'].max() - analyzer.emails['date'].min()).days)
            with col4:
                total_projects = sum(len(p) for p in analyzer.emails['projects'])
                st.metric("Projects Mentioned", total_projects)

            # Timeline
            st.subheader("Email Activity Timeline")
            timeline_fig = analyzer.visualize_timeline()
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Entity statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top Topics")
                all_topics = []
                for topics in analyzer.emails['topics']:
                    all_topics.extend(topics)
                topic_counts = Counter(all_topics).most_common(10)

                topic_df = pd.DataFrame(topic_counts, columns=['Topic', 'Count'])
                fig = px.bar(topic_df, x='Count', y='Topic', orientation='h')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Top Organizations")
                all_orgs = []
                for orgs in analyzer.emails['organizations']:
                    all_orgs.extend(orgs)
                org_counts = Counter(all_orgs).most_common(10)

                org_df = pd.DataFrame(org_counts, columns=['Organization', 'Count'])
                fig = px.bar(org_df, x='Count', y='Organization', orientation='h')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("📚 Discovered Stories")

            if analyzer.stories:
                for idx, story in enumerate(analyzer.stories, 1):
                    st.markdown(f"""
                    <div class="story-card">
                        <h3>Story #{idx}: {story['title']}</h3>
                        <p><strong>Type:</strong> {story['type'].replace('_', ' ').title()}</p>
                        <p><strong>Importance Score:</strong> {story['importance_score']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📧 Emails", story['email_count'])
                    with col2:
                        st.metric("👥 Participants", len(story['participants']))
                    with col3:
                        st.metric("📅 Duration", f"{story['duration_days']} days")

                    st.write("**Summary:**", story['summary'])

                    with st.expander("View Details"):
                        st.write("**Organizations Involved:**", ', '.join(story['organizations'][:5]))

                        st.write("**Timeline Sample:**")
                        timeline_df = pd.DataFrame(story['timeline'][:5])
                        if 'date' in timeline_df.columns:
                            timeline_df['date'] = pd.to_datetime(timeline_df['date']).dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(timeline_df)

                    st.write("---")
            else:
                st.info("Click 'Generate Analysis' to discover stories")

        with tab3:
            st.header("Entity Network Visualization")

            if st.button("Generate Network"):
                with st.spinner("Building network..."):
                    network_fig = analyzer.visualize_entity_relationships()
                    st.plotly_chart(network_fig, use_container_width=True)

                if analyzer.entity_graph:
                    st.subheader("Network Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Nodes", analyzer.entity_graph.number_of_nodes())
                    with col2:
                        st.metric("Total Edges", analyzer.entity_graph.number_of_edges())
                    with col3:
                        st.metric("Network Density", f"{nx.density(analyzer.entity_graph):.3f}")

    else:
        st.info("👆 Please upload a JSON file to begin analysis")

        with st.expander("📋 Expected JSON Format"):
            st.code('''
{
  "to": "jeff.dasovich@enron.com",
  "from": "frank.vickers@enron.com",
  "date": "14.04.2000 09:58:00",
  "subject": "Re: Project Boomerang",
  "summary": "Email summary...",
  "tone_analysis": "Collaborative/Professional",
  "classification": "Internal Communication",
  "entities": {
    "people": ["John Doe", "Jane Smith"],
    "organizations": ["EES", "FERC"],
    "locations": ["California"],
    "projects": ["Project Boomerang"],
    "topics": ["energy trading", "gas storage"]
  }
}
            ''', language='json')


if __name__ == "__main__":
    main()
