import streamlit as st
import json
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from collections import Counter, defaultdict
import warnings
import requests

from src.config.config import PROCESSED_JSON_OUTPUT_100

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔍 Enron Email Analysis Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #d62728;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #d62728;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class EnronDashboard:
    def __init__(self):
        self.data = None
        self.emails_df = None
        self.relationships = None

    @st.cache_data
    def load_data(_self):
        """Load email data"""
        try:
            with open(PROCESSED_JSON_OUTPUT_100, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            st.error("📁 Data file 'enron_full_analysis_results_100.json' not found!")
            st.info("Please ensure the file is in the same directory as this script.")
            return None

    def preprocess_data(self):
        """Convert email data to DataFrame"""
        if not self.data:
            return None

        emails = self.data.get('emails', [])
        if not emails:
            return None

        processed_emails = []
        for email in emails:
            # Parse date safely
            date_str = email.get('date', '')
            try:
                if '.' in date_str and len(date_str) > 10:
                    date_obj = datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
                else:
                    date_obj = None
            except:
                date_obj = None

            processed_emails.append({
                'email_id': email.get('email_id', 0),
                'date': date_obj,
                'subject': email.get('subject', ''),
                'sender': email.get('from', ''),
                'recipients': email.get('to', ''),
                'classification': email.get('classification', 'Unknown'),
                'tone_analysis': email.get('tone_analysis', 'Unknown'),
                'summary': email.get('summary', ''),
                'entities': email.get('entities', {}),
                'people': email.get('entities', {}).get('people', []),
                'organizations': email.get('entities', {}).get('organizations', []),
                'locations': email.get('entities', {}).get('locations', []),
                'projects': email.get('entities', {}).get('projects', [])
            })

        return pd.DataFrame(processed_emails)

    def analyze_entity_relationships(self):
        """Analyze relationships between entities"""
        if self.emails_df is None:
            return {}

        relationships = {
            'people_network': nx.Graph(),
            'org_network': nx.Graph(),
            'project_network': nx.Graph(),
            'entity_cooccurrence': defaultdict(lambda: defaultdict(int))
        }

        # Build entity co-occurrence networks
        for _, email in self.emails_df.iterrows():
            entities = email['entities']

            # People network
            people = entities.get('people', [])
            for i, person1 in enumerate(people):
                for person2 in people[i + 1:]:
                    if relationships['people_network'].has_edge(person1, person2):
                        relationships['people_network'][person1][person2]['weight'] += 1
                    else:
                        relationships['people_network'].add_edge(person1, person2, weight=1)

            # Organization network
            orgs = entities.get('organizations', [])
            for i, org1 in enumerate(orgs):
                for org2 in orgs[i + 1:]:
                    if relationships['org_network'].has_edge(org1, org2):
                        relationships['org_network'][org1][org2]['weight'] += 1
                    else:
                        relationships['org_network'].add_edge(org1, org2, weight=1)

        return relationships

    def create_network_visualization(self, graph, title, max_nodes=30):
        """Create interactive network visualization"""
        if len(graph.nodes()) == 0:
            st.warning(f"No data available for {title}")
            return None

        # Limit nodes for performance
        if len(graph.nodes()) > max_nodes:
            # Get top connected nodes
            node_degrees = dict(graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            graph = graph.subgraph([node[0] for node in top_nodes])

        # Create layout
        try:
            pos = nx.spring_layout(graph, k=2, iterations=50)
        except:
            pos = {node: (np.random.random(), np.random.random()) for node in graph.nodes()}

        # Create edges
        edge_x, edge_y = [], []
        for edge in graph.edges():
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

        # Create nodes
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            degree = graph.degree(node)
            node_text.append(f'{node}<br>Connections: {degree}')
            node_size.append(min(max(degree * 5, 10), 50))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.split()[-1] if len(node.split()) > 0 else node for node in graph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Click and drag to explore the network",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
        )

        return fig

    def render_overview(self):
        """Render overview page"""
        st.markdown('<h1 class="main-header">🔍 Enron Email Network Analysis</h1>', unsafe_allow_html=True)

        if self.emails_df is None:
            st.error("No email data available. Please check the data file.")
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📧 Total Emails", len(self.emails_df))

        with col2:
            unique_senders = self.emails_df['sender'].nunique()
            st.metric("👥 Unique Senders", unique_senders)

        with col3:
            date_range = self.emails_df['date'].dropna()
            if len(date_range) > 0:
                days_span = (date_range.max() - date_range.min()).days
                st.metric("📅 Time Span (days)", days_span)
            else:
                st.metric("📅 Time Span", "Unknown")

        with col4:
            classifications = self.emails_df['classification'].nunique()
            st.metric("📂 Email Types", classifications)

        # Overview charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 Email Classifications</div>', unsafe_allow_html=True)
            class_counts = self.emails_df['classification'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribution of Email Types"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">😊 Tone Analysis</div>', unsafe_allow_html=True)
            tone_counts = self.emails_df['tone_analysis'].value_counts()
            fig = px.bar(
                x=tone_counts.values,
                y=tone_counts.index,
                orientation='h',
                title="Email Tone Distribution",
                color=tone_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)

        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("**📈 Communication Patterns**")

            # Top senders
            top_senders = self.emails_df['sender'].value_counts().head(5)
            st.write("Top email senders:")
            for sender, count in top_senders.items():
                sender_name = sender.split('@')[0] if '@' in sender else sender
                st.write(f"• {sender_name}: {count} emails")
            st.markdown('</div>', unsafe_allow_html=True)

        with insights_col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("**🏢 Key Organizations**")

            # Extract organization mentions
            all_orgs = []
            for _, email in self.emails_df.iterrows():
                all_orgs.extend(email['organizations'])

            org_counts = Counter(all_orgs).most_common(5)
            st.write("Most mentioned organizations:")
            for org, count in org_counts:
                st.write(f"• {org}: {count} mentions")
            st.markdown('</div>', unsafe_allow_html=True)

    def render_timeline_analysis(self):
        """Render timeline analysis"""
        st.markdown('<div class="section-header">📅 Timeline Analysis</div>', unsafe_allow_html=True)

        # Filter emails with valid dates
        df_with_dates = self.emails_df.dropna(subset=['date']).copy()

        if len(df_with_dates) == 0:
            st.warning("No emails with valid dates found")
            return

        # Timeline chart
        df_with_dates['date_str'] = df_with_dates['date'].dt.strftime('%Y-%m')
        timeline_data = df_with_dates.groupby(['date_str', 'classification']).size().reset_index(name='count')

        fig = px.line(
            timeline_data,
            x='date_str',
            y='count',
            color='classification',
            title="📈 Email Volume Over Time by Classification",
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Key events timeline
        st.subheader("🔑 Key Events and Critical Emails")

        # Focus on critical classifications
        critical_emails = df_with_dates[
            df_with_dates['classification'].isin([
                'Regulatory Alert / Crisis Communication',
                'Legal/Compliance Matter',
                'Market/Trading Information'
            ])
        ].sort_values('date')

        for _, email in critical_emails.head(10).iterrows():
            with st.expander(f"📧 {email['date'].strftime('%Y-%m-%d')} - {email['subject'][:80]}..."):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**From:** {email['sender']}")
                    st.write(f"**Classification:** {email['classification']}")
                    st.write(f"**Summary:** {email['summary'][:300]}...")
                with col2:
                    st.write(f"**Tone:** {email['tone_analysis']}")
                    st.write(f"**Email ID:** {email['email_id']}")

    def render_network_analysis(self):
        """Render network analysis"""
        st.markdown('<div class="section-header">🕸️ Relationship Networks</div>', unsafe_allow_html=True)

        # Analyze relationships
        relationships = self.analyze_entity_relationships()

        # People network
        if len(relationships['people_network'].nodes()) > 0:
            st.subheader("👥 People Connection Network")
            fig = self.create_network_visualization(
                relationships['people_network'],
                "People mentioned together in emails"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Network statistics
                people_net = relationships['people_network']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 People", len(people_net.nodes()))
                with col2:
                    st.metric("🔗 Connections", len(people_net.edges()))
                with col3:
                    if len(people_net.nodes()) > 0:
                        density = nx.density(people_net)
                        st.metric("📊 Network Density", f"{density:.3f}")

        # Organization network
        if len(relationships['org_network'].nodes()) > 0:
            st.subheader("🏢 Organization Network")
            fig = self.create_network_visualization(
                relationships['org_network'],
                "Organizations mentioned together"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Communication patterns
        st.subheader("📧 Email Communication Patterns")

        # Create sender-recipient network
        comm_network = nx.DiGraph()
        for _, email in self.emails_df.iterrows():
            sender = email['sender']
            if '@' in sender:
                sender = sender.split('@')[0]

            recipients = str(email['recipients']).split(',')
            for recipient in recipients[:3]:  # Limit to first 3 recipients
                if '@' in recipient:
                    recipient = recipient.split('@')[0].strip()
                    if sender != recipient and recipient:
                        if comm_network.has_edge(sender, recipient):
                            comm_network[sender][recipient]['weight'] += 1
                        else:
                            comm_network.add_edge(sender, recipient, weight=1)

        if len(comm_network.nodes()) > 2:
            fig = self.create_network_visualization(
                comm_network,
                "Email Communication Flow"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    def render_entity_analysis(self):
        """Render entity analysis"""
        st.markdown('<div class="section-header">🏢 Entity Deep Dive</div>', unsafe_allow_html=True)

        # Entity frequency analysis
        entity_types = ['people', 'organizations', 'locations', 'projects']

        tabs = st.tabs([f"👥 People", f"🏢 Organizations", f"📍 Locations", f"🚀 Projects"])

        for i, entity_type in enumerate(entity_types):
            with tabs[i]:
                # Collect all entities of this type
                all_entities = []
                for _, email in self.emails_df.iterrows():
                    entities = email['entities'].get(entity_type, [])
                    all_entities.extend(entities)

                if all_entities:
                    entity_counts = Counter(all_entities)
                    top_entities = entity_counts.most_common(20)

                    # Create bar chart
                    if top_entities:
                        df_entities = pd.DataFrame(top_entities, columns=['Entity', 'Frequency'])

                        fig = px.bar(
                            df_entities,
                            x='Frequency',
                            y='Entity',
                            orientation='h',
                            title=f"Most Mentioned {entity_type.title()}",
                            color='Frequency',
                            color_continuous_scale='blues'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Show entity details
                        st.subheader("📋 Entity Details")
                        selected_entity = st.selectbox(
                            f"Select {entity_type[:-1]} to explore:",
                            [entity[0] for entity in top_entities[:10]]
                        )

                        if selected_entity:
                            # Find emails mentioning this entity
                            related_emails = self.emails_df[
                                self.emails_df[entity_type].apply(
                                    lambda x: selected_entity in x if isinstance(x, list) else False
                                )
                            ]

                            st.write(f"**{selected_entity}** appears in **{len(related_emails)} emails**")

                            # Show sample emails
                            for _, email in related_emails.head(3).iterrows():
                                with st.expander(f"📧 {email['subject'][:60]}..."):
                                    st.write(f"**Date:** {email['date']}")
                                    st.write(f"**From:** {email['sender']}")
                                    st.write(f"**Summary:** {email['summary'][:200]}...")
                else:
                    st.info(f"No {entity_type} found in the email data")

    def render_search_explorer(self):
        """Render search and exploration interface"""
        st.markdown('<div class="section-header">🔍 Email Explorer & Search</div>', unsafe_allow_html=True)

        # Search filters
        col1, col2, col3 = st.columns(3)

        with col1:
            search_term = st.text_input("🔎 Search in summaries and subjects:")
            classification_filter = st.selectbox(
                "📂 Filter by classification:",
                ['All'] + sorted(self.emails_df['classification'].unique().tolist())
            )

        with col2:
            tone_filter = st.selectbox(
                "😊 Filter by tone:",
                ['All'] + sorted(self.emails_df['tone_analysis'].unique().tolist())
            )

            sender_filter = st.selectbox(
                "👤 Filter by sender:",
                ['All'] + sorted([s for s in self.emails_df['sender'].unique() if s])[:20]
            )

        with col3:
            # Date range filter
            date_range = self.emails_df['date'].dropna()
            if len(date_range) > 0:
                start_date = st.date_input(
                    "📅 Start date:",
                    value=date_range.min().date(),
                    min_value=date_range.min().date(),
                    max_value=date_range.max().date()
                )
                end_date = st.date_input(
                    "📅 End date:",
                    value=date_range.max().date(),
                    min_value=date_range.min().date(),
                    max_value=date_range.max().date()
                )

        # Apply filters
        filtered_df = self.emails_df.copy()

        if search_term:
            mask = (
                    filtered_df['summary'].str.contains(search_term, case=False, na=False) |
                    filtered_df['subject'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        if classification_filter != 'All':
            filtered_df = filtered_df[filtered_df['classification'] == classification_filter]

        if tone_filter != 'All':
            filtered_df = filtered_df[filtered_df['tone_analysis'] == tone_filter]

        if sender_filter != 'All':
            filtered_df = filtered_df[filtered_df['sender'] == sender_filter]

        # Date filter
        if len(date_range) > 0:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= start_date) &
                (filtered_df['date'].dt.date <= end_date)
                ]

        st.write(f"**Found {len(filtered_df)} emails matching your criteria**")

        # Display results
        if len(filtered_df) > 0:
            # Sort by date (newest first)
            filtered_df = filtered_df.sort_values('date', ascending=False, na_position='last')

            st.subheader("📧 Search Results")

            for _, email in filtered_df.head(15).iterrows():
                with st.expander(
                        f"📧 [{email['classification']}] {email['subject'][:80]}..." +
                        (f" - {email['date'].strftime('%Y-%m-%d')}" if pd.notna(email['date']) else "")
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**From:** {email['sender']}")
                        st.write(f"**To:** {str(email['recipients'])[:100]}...")
                        st.write(f"**Summary:** {email['summary']}")

                        # Show entities
                        if email['people']:
                            st.write(f"**👥 People:** {', '.join(email['people'][:5])}")
                        if email['organizations']:
                            st.write(f"**🏢 Organizations:** {', '.join(email['organizations'][:5])}")
                        if email['projects']:
                            st.write(f"**🚀 Projects:** {', '.join(email['projects'][:3])}")

                    with col2:
                        st.write(f"**📧 ID:** {email['email_id']}")
                        st.write(f"**😊 Tone:** {email['tone_analysis']}")
                        st.write(f"**📂 Type:** {email['classification']}")
        else:
            st.info("No emails match your search criteria. Try adjusting the filters.")


def main():
    """Main application"""
    dashboard = EnronDashboard()

    # Sidebar
    st.sidebar.title("🔍 Enron Email Analysis")
    st.sidebar.markdown("---")

    # Load data
    dashboard.data = dashboard.load_data()
    if dashboard.data is None:
        st.stop()

    # Preprocess data
    dashboard.emails_df = dashboard.preprocess_data()
    if dashboard.emails_df is None:
        st.error("Failed to process email data")
        st.stop()

    # Navigation
    pages = {
        "📊 Overview": dashboard.render_overview,
        "📅 Timeline Analysis": dashboard.render_timeline_analysis,
        "🕸️ Network Analysis": dashboard.render_network_analysis,
        "🏢 Entity Analysis": dashboard.render_entity_analysis,
        "🔍 Email Explorer": dashboard.render_search_explorer
    }

    selected_page = st.sidebar.selectbox("Choose Analysis", list(pages.keys()))

    # Sidebar stats
    st.sidebar.markdown("### 📊 Dataset Summary")
    st.sidebar.metric("Total Emails", len(dashboard.emails_df))
    st.sidebar.metric("Unique Senders", dashboard.emails_df['sender'].nunique())

    date_range = dashboard.emails_df['date'].dropna()
    if len(date_range) > 0:
        st.sidebar.write("**📅 Date Range:**")
        st.sidebar.write(f"{date_range.min().strftime('%Y-%m-%d')} to {date_range.max().strftime('%Y-%m-%d')}")

    # Key entities preview
    st.sidebar.markdown("### 🔑 Key Players")
    all_people = []
    for _, email in dashboard.emails_df.iterrows():
        all_people.extend(email['people'])

    if all_people:
        top_people = Counter(all_people).most_common(5)
        for person, count in top_people:
            st.sidebar.write(f"• {person}: {count}")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **About this Analysis**\n\n"
        "This dashboard analyzes the Enron email dataset to uncover "
        "relationships, communication patterns, and key insights from "
        "the corporate communications during the company's crisis period."
    )

    # Render selected page
    pages[selected_page]()


if __name__ == "__main__":
    main()