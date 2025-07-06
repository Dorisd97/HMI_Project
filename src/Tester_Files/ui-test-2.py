import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from minisom import MiniSom
import numpy as np
from collections import Counter
from pyvis.network import Network
import streamlit.components.v1 as components
import os

from src.config.config import PNG_IMAGE, CACHED_CLUSTER_STORIES, CACHED_STORIES_PATH, THEMATIC_STORIES

# --- Page Configuration ---
st.set_page_config(
    page_title="The Enron Story: A Data-Driven Narrative",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Custom CSS for the Light & Colorful Theme ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


# --- Data Loading and Processing Functions (with Caching for performance) ---

@st.cache_data
def load_conversation_data():
    """Loads and prepares the data from cached_cluster_stories_less.json."""
    with open(CACHED_CLUSTER_STORIES, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.rename(columns={'title': 'topic', 'summary': 'summary', 'email_count': 'email_count'}, inplace=True)
    return df


@st.cache_data
def load_narrative_data():
    """Parses the thematic stories from the text file into a dictionary."""
    with open(THEMATIC_STORIES, 'r', encoding='utf-8') as f:
        content = f.read()

    stories_raw = re.split(r'\n📚 Theme-\d+\n============================================================\n', content)
    stories = {}
    for story_text in stories_raw:
        if "Title:" in story_text:
            try:
                title = re.search(r'Title: (.*?)\n', story_text).group(1).strip()
                # Clean up the body text
                body = story_text.split(title, 1)[1].strip()
                stories[title] = body
            except (AttributeError, IndexError):
                continue
    return stories


@st.cache_data
def load_activity_spike_data():
    """Loads and filters activity burst data from cached_stories_less.json."""
    with open(CACHED_STORIES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    activity_data = [item for item in data if item.get('type') == 'activity_burst']
    for item in activity_data:
        # Extract date for sorting
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', item['title'])
        item['date'] = date_match.group(0) if date_match else '2000-01-01'
    # Sort by date
    activity_data.sort(key=lambda x: x['date'])
    return activity_data


# --- Visualization and Analysis Functions ---

@st.cache_data
def generate_timeline_plot(spike_data):
    """Creates an interactive timeline of key events."""
    if not spike_data:
        return None
    df_spikes = pd.DataFrame(spike_data)
    df_spikes['date'] = pd.to_datetime(df_spikes['date'])
    df_spikes['event_name'] = df_spikes['title'].apply(lambda x: x.split(': ')[1])

    fig = px.scatter(df_spikes,
                     x='date',
                     y='email_count',
                     size='email_count',
                     color='event_name',
                     hover_name='event_name',
                     hover_data={'date': True, 'email_count': True, 'duration_days': True},
                     title="Key Event Timeline: Email Activity Spikes",
                     labels={'date': 'Date', 'email_count': 'Email Volume', 'event_name': 'Event'},
                     template="plotly_white")

    fig.update_layout(
        xaxis_title="Date of Event",
        yaxis_title="Total Emails in Spike",
        legend_title="Events",
        showlegend=False
    )
    return fig


@st.cache_data
def generate_som_map(_df):
    """Generates a Self-Organizing Map (Kohonen Map) of conversation topics."""
    df_filtered = _df[_df['summary'].str.strip() != '']
    summaries = df_filtered['summary'].tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    data = vectorizer.fit_transform(summaries).toarray()
    feature_names = vectorizer.get_feature_names_out()

    som_x, som_y = 10, 10
    som = MiniSom(som_x, som_y, data.shape[1], sigma=1.5, learning_rate=0.5,
                  neighborhood_function='gaussian', random_seed=42)
    som.random_weights_init(data)
    som.train_random(data, 1000)

    win_map = som.win_map(data)

    heatmap_data = np.zeros((som_x, som_y))
    hover_text = [['' for _ in range(som_y)] for _ in range(som_x)]

    for position, winners in win_map.items():
        x, y = position
        heatmap_data[x, y] = len(winners)

        # Get top keywords for this neuron
        neuron_weights = som.get_weights()[x, y]
        top_indices = np.argsort(neuron_weights)[-5:]
        top_keywords = [feature_names[i] for i in reversed(top_indices)]
        hover_text[x][y] = f"<b>Topics:</b> {len(winners)}<br><b>Keywords:</b><br>" + "<br>".join(top_keywords)

    fig = px.imshow(heatmap_data,
                    title="Topic Map: Visualizing Conversation Clusters",
                    labels=dict(x="Topic Group X", y="Topic Group Y", color="Density"),
                    color_continuous_scale=px.colors.sequential.Viridis,
                    template="plotly_white")

    fig.update_traces(hovertemplate='%{customdata}<extra></extra>', customdata=hover_text)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


@st.cache_data
def generate_network_graph(burst_data):
    """Creates an interactive Pyvis network graph for an activity spike."""
    participants = [p.strip() for sublist in burst_data.get('participants', []) for p in sublist.split(',') if
                    '@enron.com' in p]
    organizations = [o.strip() for o in burst_data.get('organizations', [])]

    # We need connections. Let's assume the first participant is the sender to all others.
    if not participants:
        return None

    sender = participants[0]
    receivers = participants[1:] + organizations

    # Count frequencies for node sizing
    node_counts = Counter(participants + organizations)

    net = Network(height='600px', width='100%', bgcolor='#F8F8F8', font_color='#333333', notebook=True,
                  cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    # Add nodes
    all_nodes = list(set([sender] + receivers))
    for node in all_nodes:
        size = 10 + node_counts.get(node, 1) * 2
        if '@enron.com' in node:
            title = f"Person: {node}"
            color = '#0072B2'  # Blue for people
            net.add_node(node, label=node.split('@')[0], title=title, size=size, color=color)
        else:
            title = f"Organization: {node}"
            color = '#D55E00'  # Orange for orgs
            net.add_node(node, label=node, title=title, size=size, color=color)

    # Add edges
    for r in receivers:
        if sender != r:
            net.add_edge(sender, r)

    # Save to a temporary file
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = f"temp/network_{burst_data['title'].replace(':', '').replace(' ', '_')}.html"
    net.save_graph(file_path)
    return file_path


def highlight_entities(text, entities):
    """Highlights a list of entities within a body of text."""
    # Sort entities by length, longest first, to avoid partial matches (e.g., 'Lay' inside 'Layton')
    sorted_entities = sorted(entities, key=len, reverse=True)
    for entity in sorted_entities:
        # Use regex for case-insensitive replacement, but keep original casing
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        text = pattern.sub(f"<span class='entity-highlight'>{entity}</span>", text)
    return text


# --- Load all data at the start ---
df = load_conversation_data()
narratives = load_narrative_data()
activity_spikes = load_activity_spike_data()
activity_spike_titles = {spike['title']: spike for spike in activity_spikes}
# A predefined list of key entities for highlighting
KNOWN_ENTITIES = ["Enron", "Dynegy", "FERC", "California", "Lay", "Skilling", "Fastow", "Andersen", "EES",
                  "EnronOnline"]

# --- Sidebar Navigation ---
st.sidebar.image(PNG_IMAGE, use_column_width=True)
st.sidebar.title("The Enron Story")
st.sidebar.markdown("---")
st.sidebar.markdown("""
An interactive exploration of the Enron email dataset. Uncover the hidden stories through data visualization and narrative analysis.
""")
page = st.sidebar.radio(
    "Select a Chapter",
    ("Dashboard", "Investigation Files", "Event Timeline"),
    captions=["Big Picture Overview", "Explore Narratives", "Analyze Key Events"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app visualizes pre-processed summaries and stories from the Enron email corpus.")


# --- Page Implementations ---

def dashboard_page():
    st.title("📖 The Enron Story: Main Dashboard")
    st.markdown(
        "A high-level view of the dataset, highlighting key periods of activity and dominant topics of conversation.")

    col1, col2 = st.columns([1.5, 2])

    with col1:
        st.subheader("Topic Landscape")
        st.markdown(
            "This Self-Organizing Map (SOM) clusters conversations by content. Conversations with similar language appear closer together. Darker cells indicate a higher density of conversations on that topic.")
        som_fig = generate_som_map(df)
        if som_fig:
            st.plotly_chart(som_fig, use_container_width=True)
        else:
            st.warning("Could not generate the topic map.")

    with col2:
        st.subheader("Timeline of Major Events")
        st.markdown(
            "This timeline plots high-activity periods, or 'spikes', in email traffic. The size of each point corresponds to the volume of emails. Click on a point to zoom or hover to see details.")
        timeline_fig = generate_timeline_plot(activity_spikes)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.warning("Could not generate the event timeline.")


def investigation_files_page():
    st.title("🗂️ Investigation Files")
    st.markdown("Explore the automatically generated narratives and the underlying conversation data.")

    tab1, tab2 = st.tabs(["Generated Narratives", "Conversation Dossiers"])

    with tab1:
        st.subheader("Select a Narrative to Read")
        selected_narrative_title = st.selectbox(
            "Choose a narrative:",
            options=list(narratives.keys()),
            index=0
        )
        if selected_narrative_title:
            story_content = narratives[selected_narrative_title]
            highlighted_story = highlight_entities(story_content, KNOWN_ENTITIES)

            st.markdown(f"### {selected_narrative_title}")
            st.markdown(f"<div class='storybook-box'>{highlighted_story}</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Search Conversation Summaries")
        search_query = st.text_input(
            "Search for keywords in topics or summaries (e.g., 'Dynegy', 'Merger', 'California')")

        if search_query:
            results = df[df.apply(
                lambda row: search_query.lower() in str(row['topic']).lower() or search_query.lower() in str(
                    row['summary']).lower(), axis=1)]
        else:
            results = df.sort_values('email_count', ascending=False).head(10)
            st.info("Showing the top 10 most active conversations. Use the search bar to find specific topics.")

        st.markdown(f"**Found {len(results)} matching conversation dossiers.**")

        for index, row in results.iterrows():
            with st.expander(f"**{row['topic']}** ({row['email_count']} emails)"):
                st.markdown(f"**Conversation ID:** `{row['cluster_id']}`")
                st.markdown("**Generated Summary:**")
                st.write(row['summary'])


def event_timeline_page():
    st.title("🌐 Event Timeline & Network Analysis")
    st.markdown(
        "Select a high-activity event from the timeline to analyze the key players and their communication network.")

    selected_spike_title = st.selectbox(
        "Select an Event Spike to Analyze:",
        options=list(activity_spike_titles.keys())
    )

    if selected_spike_title:
        spike_data = activity_spike_titles[selected_spike_title]
        st.header(f"Analysis of: {spike_data['title']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Email Volume", f"{spike_data['email_count']:,}")
        col2.metric("Duration", f"{spike_data['duration_days']} Days")
        col3.metric("Avg. Emails/Day", f"{int(spike_data['email_count'] / spike_data.get('duration_days', 1)):,}")

        st.subheader("Investigative Narrative")
        summary_html = spike_data['summary'].replace('\n', '<br>')
        st.markdown(f"<div class='summary-box'>{summary_html}</div>", unsafe_allow_html=True)

        st.subheader("Communication Network Graph")
        st.markdown(
            "This graph shows the connections between participants and organizations during the event. Hover over nodes for details, and drag them to explore the network. The central node is assumed to be the primary sender in the spike.")

        network_path = generate_network_graph(spike_data)
        if network_path and os.path.exists(network_path):
            with open(network_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=610)
        else:
            st.warning("Could not generate a network graph for this event (insufficient participant data).")


# --- Main Router ---
if page == "Dashboard":
    dashboard_page()
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Event Timeline":
    event_timeline_page()