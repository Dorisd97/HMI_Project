import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter, defaultdict
import networkx as nx
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from src.config.config import THEMATIC_STORIES, CACHED_CLUSTER_STORIES, CACHED_STORIES_PATH

# Page configuration
st.set_page_config(
    page_title="Enron Email Analysis Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .story-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all data files"""
    try:
        # Load thematic stories (main focus)
        with open(THEMATIC_STORIES, 'r', encoding='utf-8') as f:
            thematic_content = f.read()

        # Load cluster stories
        with open(CACHED_CLUSTER_STORIES, 'r', encoding='utf-8') as f:
            cluster_stories = json.load(f)

        # Load activity burst data
        with open(CACHED_STORIES_PATH, 'r', encoding='utf-8') as f:
            activity_data = json.load(f)

        return thematic_content, cluster_stories, activity_data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None


def parse_thematic_stories(content):
    """Parse thematic stories from text content"""
    stories = []

    # Split by theme markers
    theme_sections = content.split('📚 Theme-')[1:]  # Skip the first empty split

    for section in theme_sections:
        lines = section.strip().split('\n')
        if len(lines) < 2:
            continue

        # Extract theme number
        theme_num = lines[0].split('\n')[0].strip()

        # Extract title
        title_line = next((line for line in lines if line.startswith(' Title:')), '')
        title = title_line.replace(' Title:', '').strip() if title_line else f"Theme {theme_num}"

        # Extract content
        content_start = next((i for i, line in enumerate(lines) if 'Title:' in line), 0) + 1
        story_content = '\n'.join(lines[content_start:]).strip()

        stories.append({
            'theme_number': theme_num,
            'title': title,
            'content': story_content
        })

    return stories


def create_timeline_chart(activity_data):
    """Create timeline visualization"""
    if not activity_data:
        return None

    # Extract timeline data from activity burst
    timeline_data = []
    for item in activity_data:
        if item['type'] == 'activity_burst':
            date_range = item.get('date_range', [])
            if len(date_range) >= 2:
                try:
                    start_date = pd.to_datetime(date_range[0])
                    end_date = pd.to_datetime(date_range[1])
                    timeline_data.append({
                        'Date': start_date,
                        'Event': item['title'],
                        'Email_Count': item['email_count'],
                        'Duration': item.get('duration_days', 1)
                    })
                except Exception as e:
                    # Skip invalid dates
                    continue

    if not timeline_data:
        return None

    df = pd.DataFrame(timeline_data)

    # Create the plot with error handling
    try:
        fig = px.scatter(df, x='Date', y='Email_Count',
                         size='Duration', hover_data=['Event'],
                         title='Enron Email Activity Timeline',
                         labels={'Email_Count': 'Number of Emails'})

        fig.update_layout(height=400)
        return fig
    except Exception as e:
        # Fallback to bar chart if scatter fails
        try:
            fig = px.bar(df, x='Date', y='Email_Count',
                         hover_data=['Event', 'Duration'],
                         title='Enron Email Activity Timeline',
                         labels={'Email_Count': 'Number of Emails'})
            fig.update_layout(height=400)
            return fig
        except Exception:
            return None


def create_network_graph(cluster_stories):
    """Create network graph of organizations"""
    if not cluster_stories:
        return None

    # Extract organizations from cluster stories
    all_orgs = []
    for story in cluster_stories:
        summary = story.get('summary', '')
        # Simple extraction of organization names (can be improved)
        org_mentions = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', summary)
        all_orgs.extend(org_mentions)

    # Count organization mentions
    org_counts = Counter(all_orgs)
    top_orgs = dict(org_counts.most_common(20))

    # Create network graph
    G = nx.Graph()
    for org, count in top_orgs.items():
        G.add_node(org, size=count)

    # Add edges between frequently co-mentioned organizations
    org_list = list(top_orgs.keys())
    for i, org1 in enumerate(org_list):
        for org2 in org_list[i + 1:]:
            # Simple co-occurrence check
            co_occurrence = sum(1 for story in cluster_stories
                                if org1.lower() in story.get('summary', '').lower()
                                and org2.lower() in story.get('summary', '').lower())
            if co_occurrence > 1:
                G.add_edge(org1, org2, weight=co_occurrence)

    if len(G.nodes()) == 0:
        return None

    # Create plotly network graph
    pos = nx.spring_layout(G, k=1, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node].get('size', 1) * 2)

    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="middle center",
                            hoverinfo='text',
                            marker=dict(size=node_size,
                                        color='lightblue',
                                        line=dict(width=2, color='darkblue')))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text='Organization Network', font_size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Network of organizations mentioned in Enron emails",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="grey", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig


def extract_entities_advanced(text):
    """Extract entities from text using regex patterns"""
    # Companies (with Corp, Inc, LLC, etc.)
    companies = re.findall(
        r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:Corp|Corporation|Inc|LLC|Company|Co\.|Ltd|Limited|Energy|Power|Gas|Electric|Trading))\b',
        text)

    # General entities (capitalized words/phrases)
    entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}\b', text)

    # Filter out common stop words
    stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For',
                  'Of', 'With', 'By', 'From', 'About', 'Into', 'Through', 'During', 'Before', 'After', 'Above', 'Below',
                  'Up', 'Down', 'Out', 'Off', 'Over', 'Under', 'Again', 'Further', 'Then', 'Once', 'Here', 'There',
                  'When', 'Where', 'Why', 'How', 'All', 'Any', 'Both', 'Each', 'Few', 'More', 'Most', 'Other', 'Some',
                  'Such', 'No', 'Nor', 'Not', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very', 'Can', 'Will', 'Just',
                  'Should', 'Now', 'First', 'Second', 'Third', 'Last', 'Next', 'Previous', 'Following', 'Main', 'Key',
                  'Important', 'Major', 'Minor', 'New', 'Old', 'Current', 'Recent', 'Future', 'Past', 'Present'}

    all_entities = list(set(companies + entities))
    filtered_entities = [entity for entity in all_entities if entity not in stop_words and len(entity) > 2]

    return filtered_entities


def categorize_theme(title, content):
    """Categorize themes based on content"""
    text = f"{title} {content}".lower()

    categories = {
        'Mergers & Acquisitions': ['merger', 'acquisition', 'dynegy', 'deal', 'takeover', 'buyout'],
        'Financial Issues': ['financial', 'credit', 'debt', 'payment', 'money', 'bankruptcy', 'loss'],
        'Legal Matters': ['legal', 'lawsuit', 'court', 'settlement', 'litigation', 'regulation'],
        'Energy Trading': ['gas', 'power', 'energy', 'trading', 'pipeline', 'electricity'],
        'Corporate Governance': ['board', 'executive', 'ceo', 'management', 'governance', 'directors'],
        'Regulatory Affairs': ['ferc', 'sec', 'regulatory', 'commission', 'compliance'],
        'Network & Technology': ['network', 'system', 'technology', 'online', 'platform'],
        'General Business': []  # Default category
    }

    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category

    return 'General Business'


def build_thematic_network(thematic_stories):
    """Build network graph from thematic stories"""
    G = nx.Graph()
    theme_entities = {}

    # Process each thematic story
    for story in thematic_stories:
        theme_id = f"Theme-{story['theme_number']}"
        title = story.get('title', '')
        content = story.get('content', '')

        # Extract entities from title and content
        entities = extract_entities_advanced(f"{title} {content}")
        entities = entities[:8]  # Limit to top 8 entities per theme

        # Categorize theme
        category = categorize_theme(title, content)

        # Store for later use
        theme_entities[theme_id] = {
            'entities': entities,
            'title': title,
            'category': category,
            'word_count': len(content.split())
        }

        # Add theme node
        G.add_node(theme_id,
                   node_type='theme',
                   title=title,
                   category=category,
                   word_count=len(content.split()),
                   size=min(len(content.split()) / 50, 30))

        # Add entity nodes and connections
        for entity in entities:
            if not G.has_node(entity):
                G.add_node(entity,
                           node_type='entity',
                           themes=[],
                           size=5)

            G.nodes[entity]['themes'].append(theme_id)
            G.add_edge(theme_id, entity, edge_type='theme_entity')

    # Add edges between themes that share entities
    themes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'theme']

    for i, theme1 in enumerate(themes):
        for theme2 in themes[i + 1:]:
            # Check for shared entities
            theme1_entities = set(theme_entities[theme1]['entities'])
            theme2_entities = set(theme_entities[theme2]['entities'])

            shared_entities = theme1_entities.intersection(theme2_entities)

            if shared_entities:
                weight = len(shared_entities)
                G.add_edge(theme1, theme2,
                           edge_type='theme_theme',
                           weight=weight,
                           shared_entities=list(shared_entities))

    return G, theme_entities


def create_thematic_network_plotly(G, theme_entities):
    """Create interactive network visualization using Plotly"""
    if not G.nodes():
        return None

    # Create layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        pos = nx.random_layout(G, seed=42)

    # Separate nodes by type
    theme_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'theme']
    entity_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'entity']

    # Prepare edges
    theme_theme_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'theme_theme']
    theme_entity_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == 'theme_entity']

    # Create edge traces
    edge_traces = []

    # Theme-theme edges (red)
    edge_x_tt = []
    edge_y_tt = []
    for edge in theme_theme_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x_tt.extend([x0, x1, None])
        edge_y_tt.extend([y0, y1, None])

    if edge_x_tt:
        edge_trace_tt = go.Scatter(x=edge_x_tt, y=edge_y_tt,
                                   line=dict(width=2, color='#FF6B6B'),
                                   hoverinfo='none',
                                   mode='lines',
                                   name='Theme Connections')
        edge_traces.append(edge_trace_tt)

    # Theme-entity edges (teal)
    edge_x_te = []
    edge_y_te = []
    for edge in theme_entity_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x_te.extend([x0, x1, None])
        edge_y_te.extend([y0, y1, None])

    if edge_x_te:
        edge_trace_te = go.Scatter(x=edge_x_te, y=edge_y_te,
                                   line=dict(width=1, color='#4ECDC4'),
                                   hoverinfo='none',
                                   mode='lines',
                                   name='Theme-Entity Links')
        edge_traces.append(edge_trace_te)

    # Create node traces
    node_traces = []

    # Theme nodes (red)
    if theme_nodes:
        theme_x = [pos[node][0] for node in theme_nodes]
        theme_y = [pos[node][1] for node in theme_nodes]
        theme_text = [f"T-{node.split('-')[-1]}" for node in theme_nodes]
        theme_sizes = [max(G.nodes[node].get('size', 10) * 3, 15) for node in theme_nodes]
        theme_hover = [f"<b>{G.nodes[node].get('title', node)}</b><br>"
                       f"Category: {G.nodes[node].get('category', 'Unknown')}<br>"
                       f"Word Count: {G.nodes[node].get('word_count', 0)}<br>"
                       f"Connections: {G.degree(node)}"
                       for node in theme_nodes]

        theme_trace = go.Scatter(x=theme_x, y=theme_y,
                                 mode='markers+text',
                                 text=theme_text,
                                 textposition="middle center",
                                 hovertemplate='%{hovertext}<extra></extra>',
                                 hovertext=theme_hover,
                                 marker=dict(size=theme_sizes,
                                             color='#FF6B6B',
                                             line=dict(width=2, color='darkred')),
                                 name='Themes')
        node_traces.append(theme_trace)

    # Entity nodes (teal)
    if entity_nodes:
        entity_x = [pos[node][0] for node in entity_nodes]
        entity_y = [pos[node][1] for node in entity_nodes]
        entity_text = [node[:10] + '...' if len(node) > 10 else node for node in entity_nodes]
        entity_sizes = [max(G.nodes[node].get('size', 5) * 2, 8) for node in entity_nodes]
        entity_hover = [f"<b>{node}</b><br>"
                        f"Connected Themes: {G.degree(node)}<br>"
                        f"Type: Entity"
                        for node in entity_nodes]

        entity_trace = go.Scatter(x=entity_x, y=entity_y,
                                  mode='markers+text',
                                  text=entity_text,
                                  textposition="middle center",
                                  hovertemplate='%{hovertext}<extra></extra>',
                                  hovertext=entity_hover,
                                  marker=dict(size=entity_sizes,
                                              color='#4ECDC4',
                                              line=dict(width=1, color='darkcyan')),
                                  name='Entities')
        node_traces.append(entity_trace)

    # Create the figure
    fig = go.Figure(data=edge_traces + node_traces)

    # Update layout
    fig.update_layout(
        title=dict(
            text="🕸️ Thematic Stories Network<br><sub>Red = Themes, Teal = Entities</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        annotations=[
            dict(
                text=f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="grey", size=10)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor='white'
    )

    return fig
    """Create network graph for a specific cluster story"""
    if not cluster_story:
        return None

    summary = cluster_story.get('summary', '')
    title = cluster_story.get('title', '')

    # Extract entities from both summary and title
    text = f"{title} {summary}"

    # Enhanced entity extraction
    # Companies (often have Corp, Inc, LLC, Company, etc.)
    companies = re.findall(
        r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:Corp|Corporation|Inc|LLC|Company|Co\.|Ltd|Limited|Energy|Power|Gas|Electric|Trading))\b',
        text)

    # General entities (capitalized words/phrases)
    entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2}\b', text)

    # Combine and filter entities
    all_entities = list(set(companies + entities))

    # Filter out common words that aren't entities
    stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For',
                  'Of', 'With', 'By', 'From', 'About', 'Into', 'Through', 'During', 'Before', 'After', 'Above', 'Below',
                  'Up', 'Down', 'Out', 'Off', 'Over', 'Under', 'Again', 'Further', 'Then', 'Once', 'Here', 'There',
                  'When', 'Where', 'Why', 'How', 'All', 'Any', 'Both', 'Each', 'Few', 'More', 'Most', 'Other', 'Some',
                  'Such', 'No', 'Nor', 'Not', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very', 'Can', 'Will', 'Just',
                  'Should', 'Now'}

    filtered_entities = [entity for entity in all_entities if entity not in stop_words and len(entity) > 2]

    # Take top entities by frequency
    entity_counts = Counter(filtered_entities)
    top_entities = dict(entity_counts.most_common(15))  # Limit to 15 for readability

    if len(top_entities) < 2:
        return None

    # Create network graph
    G = nx.Graph()

    # Add nodes
    for entity, count in top_entities.items():
        G.add_node(entity, size=count * 5)  # Scale for visualization

    # Add edges based on co-occurrence in sentences
    sentences = re.split(r'[.!?]+', text)
    entity_list = list(top_entities.keys())

    for sentence in sentences:
        entities_in_sentence = [entity for entity in entity_list if entity.lower() in sentence.lower()]
        # Create edges between entities that appear in the same sentence
        for i, entity1 in enumerate(entities_in_sentence):
            for entity2 in entities_in_sentence[i + 1:]:
                if G.has_edge(entity1, entity2):
                    G[entity1][entity2]['weight'] += 1
                else:
                    G.add_edge(entity1, entity2, weight=1)

    if len(G.nodes()) == 0:
        return None

    # Create plotly network graph
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.random_layout(G)

    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G[edge[0]][edge[1]].get('weight', 1)
        edge_weights.extend([weight, weight, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_info = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        size = G.nodes[node].get('size', 10)
        node_size.append(max(size, 10))  # Minimum size

        # Node info for hover
        adjacencies = list(G.neighbors(node))
        node_info.append(f'{node}<br>Connections: {len(adjacencies)}<br>Mentions: {top_entities.get(node, 1)}')

    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="middle center",
                            hoverinfo='text',
                            hovertext=node_info,
                            marker=dict(size=node_size,
                                        color='lightblue',
                                        line=dict(width=2, color='darkblue')))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=f'Entity Network for Cluster {cluster_story.get("cluster_id", "")}',
                                   font_size=14),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Entities and their relationships in this cluster",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="grey", size=10)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400))

    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Enron Email Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    thematic_content, cluster_stories, activity_data = load_data()

    if not thematic_content:
        st.error("Unable to load data files. Please ensure all files are in the correct location.")
        return

    # Parse thematic stories
    thematic_stories = parse_thematic_stories(thematic_content)

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("📊 Navigation")

    # Main navigation
    main_tab = st.sidebar.radio(
        "Choose Analysis Type:",
        ["🏠 Overview", "📚 Thematic Stories", "🔍 Cluster Analysis", "📈 Activity Patterns", "🌐 Network Analysis"]
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Overview Tab
    if main_tab == "🏠 Overview":
        st.header("📋 Dataset Overview")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Thematic Stories", len(thematic_stories))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Cluster Stories", len(cluster_stories) if cluster_stories else 0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_emails = sum(story.get('email_count', 0) for story in cluster_stories) if cluster_stories else 0
            st.metric("Total Emails Analyzed", total_emails)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            activity_items = len(activity_data) if activity_data else 0
            st.metric("Activity Patterns", activity_items)
            st.markdown('</div>', unsafe_allow_html=True)

        # Timeline chart
        st.subheader("📅 Email Activity Timeline")
        timeline_fig = create_timeline_chart(activity_data)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Timeline data not available")

        # Quick insights
        st.subheader("🔍 Quick Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Email Clusters by Volume")
            if cluster_stories:
                top_clusters = sorted(cluster_stories, key=lambda x: x.get('email_count', 0), reverse=True)[:5]
                for i, cluster in enumerate(top_clusters, 1):
                    st.write(f"{i}. **{cluster.get('title', 'Untitled')}** ({cluster.get('email_count', 0)} emails)")

        with col2:
            st.markdown("#### Thematic Story Categories")
            if thematic_stories:
                theme_keywords = []
                for story in thematic_stories:
                    title = story.get('title', '')
                    # Extract key themes from titles
                    if 'merger' in title.lower():
                        theme_keywords.append('Mergers & Acquisitions')
                    elif 'financial' in title.lower() or 'credit' in title.lower():
                        theme_keywords.append('Financial Issues')
                    elif 'settlement' in title.lower() or 'legal' in title.lower():
                        theme_keywords.append('Legal Matters')
                    elif 'gas' in title.lower() or 'power' in title.lower():
                        theme_keywords.append('Energy Trading')
                    else:
                        theme_keywords.append('General Business')

                theme_counts = Counter(theme_keywords)
                for theme, count in theme_counts.most_common(5):
                    st.write(f"• **{theme}**: {count} stories")

    # Thematic Stories Tab (Main Focus)
    elif main_tab == "📚 Thematic Stories":
        st.header("📚 Thematic Story Analysis")
        st.markdown("*Deep dive into the key narratives extracted from Enron email communications*")

        # Build and display thematic network
        st.subheader("🕸️ Thematic Stories Network")
        st.markdown("*Interactive network showing relationships between themes and entities*")

        with st.spinner("Building thematic network..."):
            G, theme_entities = build_thematic_network(thematic_stories)
            network_fig = create_thematic_network_plotly(G, theme_entities)

            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)

                # Network statistics
                col1, col2, col3, col4 = st.columns(4)

                theme_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'theme']
                entity_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == 'entity']

                with col1:
                    st.metric("Total Themes", len(theme_nodes))
                with col2:
                    st.metric("Total Entities", len(entity_nodes))
                with col3:
                    st.metric("Connections", G.number_of_edges())
                with col4:
                    if len(theme_nodes) > 0:
                        avg_connections = G.number_of_edges() / len(theme_nodes)
                        st.metric("Avg Connections/Theme", f"{avg_connections:.1f}")

                # Network insights
                with st.expander("🔍 Network Insights"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Most Connected Themes:**")
                        theme_degrees = [(node, G.degree(node)) for node in theme_nodes]
                        theme_degrees.sort(key=lambda x: x[1], reverse=True)

                        for i, (theme, degree) in enumerate(theme_degrees[:5], 1):
                            title = theme_entities[theme]['title']
                            st.write(f"{i}. **{title[:50]}...** ({degree} connections)")

                    with col2:
                        st.markdown("**Most Connected Entities:**")
                        entity_degrees = [(node, G.degree(node)) for node in entity_nodes]
                        entity_degrees.sort(key=lambda x: x[1], reverse=True)

                        for i, (entity, degree) in enumerate(entity_degrees[:5], 1):
                            st.write(f"{i}. **{entity}** ({degree} themes)")

                # Category distribution
                categories = [theme_entities[theme]['category'] for theme in theme_nodes]
                category_counts = Counter(categories)

                if category_counts:
                    st.subheader("📊 Theme Categories Distribution")
                    fig_cat = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="Distribution of Thematic Categories"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Unable to generate thematic network - insufficient data")

        st.markdown("---")

        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search stories by keyword:", placeholder="e.g., Dynegy, merger, financial")
        with col2:
            sort_by = st.selectbox("Sort by:", ["Theme Number", "Title Length", "Alphabetical"])

        # Filter stories
        filtered_stories = thematic_stories
        if search_term:
            filtered_stories = [story for story in thematic_stories
                                if search_term.lower() in story.get('title', '').lower()
                                or search_term.lower() in story.get('content', '').lower()]

        # Sort stories
        if sort_by == "Theme Number":
            filtered_stories = sorted(filtered_stories,
                                      key=lambda x: int(x.get('theme_number', 0)) if x.get('theme_number',
                                                                                           '').isdigit() else 0)
        elif sort_by == "Title Length":
            filtered_stories = sorted(filtered_stories, key=lambda x: len(x.get('title', '')), reverse=True)
        else:  # Alphabetical
            filtered_stories = sorted(filtered_stories, key=lambda x: x.get('title', ''))

        # Story selection
        if filtered_stories:
            story_options = [f"Theme-{story['theme_number']}: {story['title']}" for story in filtered_stories]
            selected_story_idx = st.selectbox("Select a story to read:", range(len(story_options)),
                                              format_func=lambda x: story_options[x])

            if selected_story_idx is not None:
                story = filtered_stories[selected_story_idx]

                # Display story
                st.markdown('<div class="story-card">', unsafe_allow_html=True)
                st.markdown(f"### 📚 Theme-{story['theme_number']}")
                st.markdown(f"**{story['title']}**")
                st.markdown("---")
                st.markdown(story['content'])
                st.markdown('</div>', unsafe_allow_html=True)

                # Story analytics
                with st.expander("📊 Story Analytics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        word_count = len(story['content'].split())
                        st.metric("Word Count", word_count)
                    with col2:
                        # Extract entities mentioned
                        entities = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', story['content'])
                        unique_entities = len(set(entities))
                        st.metric("Entities Mentioned", unique_entities)
                    with col3:
                        # Extract dates mentioned
                        dates = re.findall(r'\b\d{4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b', story['content'])
                        st.metric("Dates Referenced", len(set(dates)))

                    # Show entities and category for this story
                    theme_id = f"Theme-{story['theme_number']}"
                    if theme_id in theme_entities:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Category:**")
                            st.write(theme_entities[theme_id]['category'])

                        with col2:
                            st.markdown("**Key Entities:**")
                            entities_list = theme_entities[theme_id]['entities']
                            if entities_list:
                                for entity in entities_list[:5]:
                                    st.write(f"• {entity}")
                            else:
                                st.write("No entities extracted")
        else:
            st.info("No stories found matching your search criteria.")

        # Stories overview
        st.subheader("📈 Stories Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Theme distribution
            theme_nums = [int(story['theme_number']) for story in thematic_stories if story['theme_number'].isdigit()]
            if theme_nums:
                fig = px.histogram(x=theme_nums, title="Distribution of Theme Numbers",
                                   labels={'x': 'Theme Number', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Story length distribution
            story_lengths = [len(story['content'].split()) for story in thematic_stories]
            fig = px.histogram(x=story_lengths, title="Story Length Distribution (Words)",
                               labels={'x': 'Word Count', 'y': 'Number of Stories'})
            st.plotly_chart(fig, use_container_width=True)

    # Cluster Analysis Tab
    elif main_tab == "🔍 Cluster Analysis":
        st.header("🔍 Email Cluster Analysis")

        if not cluster_stories:
            st.error("Cluster data not available")
            return

        # Cluster selection
        cluster_options = [f"Cluster {story['cluster_id']}: {story['title']}" for story in cluster_stories]
        selected_cluster_idx = st.selectbox("Select a cluster:", range(len(cluster_options)),
                                            format_func=lambda x: cluster_options[x])

        cluster = cluster_stories[selected_cluster_idx]

        # Display cluster details
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### Cluster {cluster['cluster_id']}: {cluster['title']}")
            st.markdown(f"**Email Count:** {cluster['email_count']}")
            st.markdown("#### Summary")
            st.markdown(cluster['summary'])

        with col2:
            st.markdown("#### Cluster Metrics")
            st.metric("Email Volume", cluster['email_count'])

            # Extract key metrics from summary
            summary_words = len(cluster['summary'].split())
            st.metric("Summary Length", f"{summary_words} words")

        # Network Graph for this specific cluster
        st.markdown("---")
        st.subheader("🌐 Entity Network for This Cluster")
        cluster_network_fig = create_cluster_network_graph(cluster)
        if cluster_network_fig:
            st.plotly_chart(cluster_network_fig, use_container_width=True)

            # Entity analysis for this cluster
            with st.expander("📊 Entity Analysis Details"):
                summary = cluster.get('summary', '')
                title = cluster.get('title', '')
                text = f"{title} {summary}"

                # Extract and display entities
                companies = re.findall(
                    r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+(?:Corp|Corporation|Inc|LLC|Company|Co\.|Ltd|Limited|Energy|Power|Gas|Electric|Trading))\b',
                    text)
                entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2}\b', text)

                all_entities = list(set(companies + entities))
                stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At',
                              'To', 'For', 'Of', 'With', 'By', 'From', 'About', 'Into', 'Through', 'During', 'Before',
                              'After', 'Above', 'Below', 'Up', 'Down', 'Out', 'Off', 'Over', 'Under', 'Again',
                              'Further', 'Then', 'Once', 'Here', 'There', 'When', 'Where', 'Why', 'How', 'All', 'Any',
                              'Both', 'Each', 'Few', 'More', 'Most', 'Other', 'Some', 'Such', 'No', 'Nor', 'Not',
                              'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very', 'Can', 'Will', 'Just', 'Should',
                              'Now'}
                filtered_entities = [entity for entity in all_entities if entity not in stop_words and len(entity) > 2]

                entity_counts = Counter(filtered_entities)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Companies Mentioned:**")
                    company_counts = Counter(companies)
                    for company, count in company_counts.most_common(10):
                        st.write(f"• {company} ({count} times)")

                with col2:
                    st.markdown("**Key Entities:**")
                    for entity, count in entity_counts.most_common(10):
                        if entity not in companies:
                            st.write(f"• {entity} ({count} times)")
        else:
            st.info("Unable to generate network graph for this cluster - insufficient entity relationships found.")

        st.markdown("---")

        # Cluster analysis
        st.subheader("📊 Cluster Statistics")

        # Email count distribution
        email_counts = [story['email_count'] for story in cluster_stories]
        fig = px.box(y=email_counts, title="Email Count Distribution Across Clusters")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Top clusters by email volume
        st.subheader("🏆 Top Clusters by Email Volume")
        top_clusters = sorted(cluster_stories, key=lambda x: x['email_count'], reverse=True)[:10]

        df_top = pd.DataFrame([
            {'Cluster': f"Cluster {c['cluster_id']}", 'Title': c['title'], 'Email Count': c['email_count']}
            for c in top_clusters
        ])

        fig = px.bar(df_top, x='Email Count', y='Cluster', orientation='h',
                     title="Top 10 Clusters by Email Volume",
                     hover_data=['Title'])
        st.plotly_chart(fig, use_container_width=True)

    # Activity Patterns Tab
    elif main_tab == "📈 Activity Patterns":
        st.header("📈 Email Activity Patterns")

        if not activity_data:
            st.error("Activity data not available")
            return

        # Activity burst analysis
        activity_burst = next((item for item in activity_data if item['type'] == 'activity_burst'), None)

        if activity_burst:
            st.subheader(f"🚀 {activity_burst['title']}")

            # Activity metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Emails", activity_burst['email_count'])
            with col2:
                st.metric("Duration (Days)", activity_burst['duration_days'])
            with col3:
                participants = len(activity_burst.get('participants', []))
                st.metric("Participants", participants)
            with col4:
                organizations = len(activity_burst.get('organizations', []))
                st.metric("Organizations", organizations)

            # Summary
            st.markdown("#### Activity Summary")
            st.markdown(activity_burst.get('summary', 'No summary available'))

            # Timeline analysis
            if 'timeline' in activity_burst:
                st.subheader("📅 Detailed Timeline")
                timeline_df = pd.DataFrame(activity_burst['timeline'])

                if not timeline_df.empty:
                    # Convert date column
                    timeline_df['date'] = pd.to_datetime(timeline_df['date'])

                    # Add a count column for visualization (each email = 1)
                    timeline_df['email_count'] = 1

                    # Check if we have the required columns
                    required_cols = ['date', 'from']
                    if all(col in timeline_df.columns for col in required_cols):
                        # Timeline chart
                        hover_data = ['subject'] if 'subject' in timeline_df.columns else None
                        fig = px.scatter(timeline_df, x='date', y='from',
                                         size='email_count',
                                         hover_data=hover_data,
                                         title="Email Timeline")
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback: simple timeline without size
                        st.write("Timeline data structure:")
                        st.dataframe(timeline_df.head())

                        if 'date' in timeline_df.columns:
                            # Group by date to show email frequency
                            daily_counts = timeline_df.groupby(timeline_df['date'].dt.date).size().reset_index()
                            daily_counts.columns = ['Date', 'Email_Count']

                            fig = px.bar(daily_counts, x='Date', y='Email_Count',
                                         title="Daily Email Volume")
                            st.plotly_chart(fig, use_container_width=True)

                    # Top senders
                    st.subheader("👥 Top Email Senders")
                    if 'from' in timeline_df.columns:
                        sender_counts = timeline_df['from'].value_counts().head(10)
                        if not sender_counts.empty:
                            fig = px.bar(x=sender_counts.values, y=sender_counts.index,
                                         orientation='h', title="Most Active Email Senders")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No sender data available")
                    else:
                        st.info("Sender information not available in timeline data")

            # Organizations network
            if activity_burst.get('organizations'):
                st.subheader("🏢 Organizations Involved")
                orgs = activity_burst['organizations']

                # Create organization frequency chart
                org_df = pd.DataFrame({'Organization': orgs})
                org_counts = org_df['Organization'].value_counts().head(15)

                fig = px.pie(values=org_counts.values, names=org_counts.index,
                             title="Organization Mentions")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity burst data available")

    # Network Analysis Tab
    elif main_tab == "🌐 Network Analysis":
        st.header("🌐 Network Analysis")

        # Organization network
        st.subheader("🏢 Organization Network")
        network_fig = create_network_graph(cluster_stories)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("Unable to generate network graph")

        # Key players analysis
        if cluster_stories:
            st.subheader("👥 Key Players Analysis")

            # Extract all organizations mentioned
            all_orgs = []
            for story in cluster_stories:
                summary = story.get('summary', '')
                orgs = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', summary)
                all_orgs.extend(orgs)

            org_counts = Counter(all_orgs)
            top_orgs = org_counts.most_common(20)

            if top_orgs:
                df_orgs = pd.DataFrame(top_orgs, columns=['Organization', 'Mentions'])

                fig = px.bar(df_orgs, x='Mentions', y='Organization',
                             orientation='h', title="Most Mentioned Organizations")
                st.plotly_chart(fig, use_container_width=True)

                # Organization details
                st.subheader("📊 Organization Details")
                selected_org = st.selectbox("Select organization for details:",
                                            [org for org, _ in top_orgs])

                if selected_org:
                    # Find stories mentioning this organization
                    related_stories = [story for story in cluster_stories
                                       if selected_org.lower() in story.get('summary', '').lower()]

                    st.write(f"**{selected_org}** appears in {len(related_stories)} cluster summaries:")
                    for story in related_stories[:5]:  # Show top 5
                        st.write(f"• Cluster {story['cluster_id']}: {story['title']}")


if __name__ == "__main__":
    main()