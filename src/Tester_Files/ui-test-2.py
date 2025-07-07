import streamlit as st
import pandas as pd
import re
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from collections import Counter
import plotly.express as px
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

from itertools import combinations

from src.config.config import PNG_IMAGE, PICKEL_FILE, PROCESSED_JSON_OUTPUT, CACHED_CLUSTER_STORIES

# --- Configuration Setup ---
# The original config import is commented out to make this script self-contained.
# from src.config.config import PROCESSED_JSON_OUTPUT, PNG_IMAGE, PICKEL_FILE, THEMATIC_STORIES

# --- Paths are defined directly here. ---
# Ensure these files are in the same directory as your script or provide the correct path.
PNG_IMAGE = PNG_IMAGE  # A placeholder name for your logo image
PICKLE_DIR = PICKEL_FILE  # Directory for .pkl files
PROCESSED_JSON_OUTPUT = PROCESSED_JSON_OUTPUT  # Required for the Dashboard page
# This JSON is the source for the new Network page and the AI Summary
THEMATIC_STORIES = CACHED_CLUSTER_STORIES

# --- Advanced Analysis Imports ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from minisom import MiniSom

    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False

# --- LangChain & Ollama Imports ---
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="The Enron Files: A Narrative Investigation",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# You can create a style.css file to customize further if needed
# local_css("style.css")


# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_precomputed_pkl(file_name):
    """Loads pre-computed data for the narrative/timeline pages."""
    path = os.path.join(PICKLE_DIR, file_name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Fail silently if optional pickle files aren't present
        return None


@st.cache_data
def load_email_dataframe(filepath=PROCESSED_JSON_OUTPUT):
    """Loads the original email JSON for the Dashboard page."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'emails' in data:
            emails = data['emails']
        elif isinstance(data, list):
            emails = data
        else:
            return None
        df = pd.DataFrame(emails)
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        df['full_text'] = df['subject'].fillna('') + ' ' + df['summary'].fillna('')
        return df
    except FileNotFoundError:
        st.error(f"The main data file '{filepath}' was not found. The 'Dashboard' page will not work.")
        return None
    except Exception as e:
        st.error(f"Error loading main data file: {e}")
        return None


@st.cache_data
def load_stories_from_json(file_path):
    """Loads the stories from the provided JSON file for the network graph."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {story['title']: story for story in data}
    except FileNotFoundError:
        st.error(f"Story file not found: '{file_path}'. The 'Entity Relationship Network' page requires this file.")
        return {}
    except Exception as e:
        st.error(f"Error loading story file '{file_path}': {e}")
        return {}


# --- LLM & VISUALIZATION FUNCTIONS ---
@st.cache_data
def generate_llm_summary(file_path):
    """Generates an executive summary using a local LLM."""
    if not LANGCHAIN_AVAILABLE:
        return "Could not generate summary: `langchain` libraries not installed."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # We read the JSON and extract summaries to create the context
            stories = json.load(f)
            context = "\n\n---\n\n".join([f"Title: {s['title']}\nSummary: {s['summary']}" for s in stories])

        llm = ChatOllama(model="mistral")
        prompt = f"""
        As a financial investigator analyzing the Enron scandal, provide a high-level executive summary based on the provided collection of email cluster summaries.
        Your summary should:
        1. Identify the main recurring themes (e.g., financial manipulation, regulatory issues, the Dynegy merger).
        2. Name the key companies and individuals who appear frequently.
        3. Describe the general timeline or progression of events if evident.
        4. Conclude with a synthesis of the key events that led to Enron's downfall, according to this text.
        Keep the summary concise and professional. CONTEXT:\n---\n{context}\n---
        """
        message = HumanMessage(content=prompt)
        response = llm([message])
        return response.content
    except Exception as e:
        return f"**Error: Could not connect to local LLM.** Ensure Ollama is running with the 'mistral' model available. Error: {e}"


def extract_keywords_from_title(title):
    stop_words = {'a', 'an', 'and', 'the', 'in', 'of', 'for', 'with', 'on', 'at', 'by', 'to', 'is', 'was', 'from',
                  'its'}
    words = re.split(r'\W+', title.lower())
    return [word for word in words if len(word) > 3 and word not in stop_words]


def highlight_text(text, terms_to_highlight, css_class):
    sorted_terms = sorted(list(set(terms_to_highlight)), key=len, reverse=True)
    for term in sorted_terms:
        clean_term = re.sub(r'\s*\(.*\)', '', term).strip()
        if not clean_term: continue
        pattern = re.compile(r'\b(' + re.escape(clean_term) + r')\b', re.IGNORECASE)
        text = pattern.sub(f"<span class='{css_class}'>\\1</span>", text)
    return text


@st.cache_data
def generate_narrative_network_graph(central_node_name: str, actors_list: list[str]) -> str | None:
    if not actors_list: return None
    net = Network(height='600px', width='100%', bgcolor='#FFFFFF', font_color='#333333', notebook=False,
                  cdn_resources='in_line')
    net.add_node(central_node_name, label=central_node_name, shape='star', size=30, color='#FFC300')
    for actor in actors_list:
        net.add_node(actor, label=actor, shape='dot', size=15)
        net.add_edge(central_node_name, actor)
    net.set_options(
        """{"physics": {"barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.1, "springLength": 250}}}""")
    return net.generate_html()


# --- FUNCTIONS FOR THE NEW ENTITY RELATIONSHIP NETWORK ---
@st.cache_data
def extract_entities_from_text(text: str) -> list[str]:
    """Extracts potential entities (keywords, emails, proper nouns) from text."""
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    keywords = [
        'pipeline statements', 'invoice', 'deal', 'EOL logic issues', 'settlements',
        'third party gas', 'volumes allocation', 'Enron Corp', 'imbalances',
        'failed desk to desk deals', 'third party deals', 'P/L', 'volumes',
        'Redwood Gas', 'allocation', 'processes tightening', 'volume management',
        'back office groups', 'payments', 'meeting with settlements', 'Wood report',
        'SOCAL imbalances', 'Dynegy', 'Northern Border Partners', 'Audit Committee',
        'SEC rules', 'SOCAL', 'BP', 'Enron'
    ]
    found_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)]
    proper_nouns = re.findall(r'\b([A-Z][a-zA-Z\s]+)\b', text)

    entities = emails + found_keywords + [pn.strip() for pn in proper_nouns]
    unique_entities = sorted(list(set(e for e in entities if len(e) > 2 and e not in ["The", "A"])))
    return unique_entities


@st.cache_data
def generate_styled_entity_network(title: str, summary: str, entities: list[str]) -> str | None:
    """Generates a styled pyvis network graph with a LIGHT theme."""
    # Updated to a light theme
    net = Network(height='600px', width='100%', bgcolor='#FFFFFF', font_color='black', notebook=False,
                  cdn_resources='in_line')

    # Central email nodes
    center_emails = ['anne.bike@enron.com', 'mike.grigsby@enron.com']
    for email in center_emails:
        net.add_node(email, label=email, shape='dot', size=15, color='#0d6efd')  # Darker blue for contrast

    # Peripheral entity nodes
    peripheral_nodes = [e for e in entities if e not in center_emails]
    for entity in peripheral_nodes:
        net.add_node(entity, label=entity, shape='dot', size=12, color='#0d6efd')
        # Connect all peripheral nodes to the central email nodes
        for email in center_emails:
            net.add_edge(entity, email, color='#adb5bd')  # Light grey for edges

    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -18000,
          "centralGravity": 0.4,
          "springLength": 200,
          "springConstant": 0.05
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      }
    }
    """)
    return net.generate_html()


# --- Functions for Dashboard ---
@st.cache_data(show_spinner=False)
def train_and_visualize_som(documents):
    if not ADVANCED_ANALYSIS_AVAILABLE: return None, None
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    data = vectorizer.fit_transform(documents).toarray()
    if data.shape[0] < 5: return None, None
    som_x = som_y = int(np.sqrt(5 * np.sqrt(data.shape[0])))
    som = MiniSom(som_x, som_y, data.shape[1], sigma=1.5, learning_rate=0.5, random_seed=42)
    som.random_weights_init(data)
    som.train_random(data, 1000, verbose=False)
    distance_map = som.distance_map()
    win_map = {}
    for i, doc_vector in enumerate(data):
        winner = som.winner(doc_vector)
        if winner not in win_map: win_map[winner] = []
        win_map[winner].append(i)

    # --- FINAL FIX: Reduced the figure size for the SOM plot even further ---
    fig, ax = plt.subplots(figsize=(6, 6))  # Changed from (8, 8) to (6, 6)

    im = ax.pcolormesh(distance_map.T, cmap='bone_r')
    fig.colorbar(im, ax=ax)
    ax.set_title('Self-Organizing Map of Email Documents')
    return fig, win_map


@st.cache_data
def perform_lda(documents, n_topics=5, n_top_words=10):
    if not ADVANCED_ANALYSIS_AVAILABLE: return [
        "Advanced analysis libraries not found. Please run `pip install scikit-learn`."]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    tf = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(tf)
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = " | ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(f"**Topic {topic_idx + 1}:** {top_words}")
    return topics


def create_cooccurrence_heatmap(df, entity_type='organizations', top_n=15):
    # This check is crucial since the column might not exist in all dataframes.
    if 'entities' not in df.columns:
        st.warning("The loaded data does not contain an 'entities' column required for this feature.")
        return None

    all_entities = df['entities'].apply(
        lambda x: x.get(entity_type, []) if isinstance(x, dict) else []).explode().dropna()
    top_entities = all_entities.value_counts().nlargest(top_n).index.tolist()
    co_matrix = pd.DataFrame(0, index=top_entities, columns=top_entities)
    for _, row in df.iterrows():
        # Ensure row['entities'] is a dict before calling .get()
        if isinstance(row['entities'], dict):
            row_entities = list(set(e for e in row['entities'].get(entity_type, []) if e in top_entities))
            for e1, e2 in combinations(row_entities, 2):
                co_matrix.loc[e1, e2] += 1
                co_matrix.loc[e2, e1] += 1
    fig = px.imshow(co_matrix, text_auto=True, aspect="auto",
                    labels=dict(x="Entity", y="Entity", color="Co-occurrence Count"),
                    title=f'Co-occurrence of Top {top_n} {entity_type.capitalize()}')
    return fig


# --- Sidebar ---
if os.path.exists(PNG_IMAGE):
    st.sidebar.image(PNG_IMAGE, use_container_width=True)
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select an Analysis View",
    ("Dashboard", "AI Summary", "Investigation Files", "Event Timeline", "Entity Relationship Network"),
    captions=["High-level Data Overview", "AI Executive Summary", "Explore Original Narratives",
              "Analyze Activity Spikes", "Visualize Story Concepts"]
)

st.sidebar.markdown("---")
st.sidebar.info("This app uses a mix of pre-computed data and live analysis. Ensure Ollama is running for AI features.")


# --- PAGE IMPLEMENTATIONS ---
def ai_summary_page():
    st.title("🤖 AI Summary")
    st.markdown("A high-level overview of the case, featuring an AI-generated summary from the thematic stories file.")
    st.header("AI-Generated Executive Summary")

    if not os.path.exists(THEMATIC_STORIES):
        st.error(f"The required data file '{THEMATIC_STORIES}' was not found.")
        return

    if 'llm_summary' not in st.session_state:
        st.session_state.llm_summary = None
    if st.session_state.llm_summary:
        st.markdown(f"<div class='llm-summary-box'>{st.session_state.llm_summary}</div>", unsafe_allow_html=True)
    else:
        st.info("Click the button below to generate the AI summary. This is a one-time operation per session.")
        if st.button("Generate AI Summary"):
            with st.spinner("Analyzing narratives with local AI model..."):
                summary = generate_llm_summary(THEMATIC_STORIES)
                st.session_state.llm_summary = summary
                st.rerun()


def dashboard_page():
    """Displays high-level stats and advanced research tools."""
    st.title("📊 Research Dashboard")
    st.markdown(
        "A deep-dive analysis of the raw email data, featuring high-level statistics and machine learning-driven insights.")

    df = load_email_dataframe()
    if df is None: return

    st.header("High-Level Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails Analyzed", len(df))
    col2.metric("Unique Senders", df['from'].nunique())
    col3.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    if 'classification' in df.columns and 'tone_analysis' in df.columns:
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.subheader("Email Classification")
            classification_counts = df['classification'].value_counts()
            fig_class = px.pie(classification_counts, values=classification_counts.values,
                               names=classification_counts.index, title="Distribution of Email Classifications")
            st.plotly_chart(fig_class, use_container_width=True)
        with col_viz2:
            st.subheader("Tone Analysis")
            tone_counts = df['tone_analysis'].value_counts()
            fig_tone = px.pie(tone_counts, values=tone_counts.values, names=tone_counts.index,
                              title="Distribution of Email Tones")
            st.plotly_chart(fig_tone, use_container_width=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.header("Advanced Research Tools")

    with st.expander("🧠 Kohonen Map (SOM) Document Clustering"):
        if st.button("Train SOM and Generate Map"):
            with st.spinner("Training SOM... This may take some time."):
                som_fig, win_map = train_and_visualize_som(df['full_text'])
            if som_fig:
                st.pyplot(som_fig)
                st.subheader("Explore Clusters")
                cluster_options = {f"Cluster at ({x}, {y}) - {len(docs)} emails": (x, y) for (x, y), docs in
                                   win_map.items()}
                selected_cluster_key = st.selectbox("Select a cluster to see its email subjects:",
                                                    options=cluster_options.keys())
                if selected_cluster_key:
                    win_coords = cluster_options[selected_cluster_key]
                    doc_indices = win_map[win_coords]
                    for idx in doc_indices[:10]: st.caption(f"- `{df.iloc[idx]['subject']}`")
            else:
                st.warning("Not enough data to generate a SOM.")

    with st.expander("📚 Latent Dirichlet Allocation (LDA) Topic Modeling"):
        num_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=5, step=1, key="lda_slider")
        with st.spinner("Finding topics..."):
            discovered_topics = perform_lda(df['full_text'], n_topics=num_topics)
            st.subheader(f"Discovered Topics")
            for topic in discovered_topics: st.markdown(topic)

    with st.expander("🔗 Entity Co-occurrence Heatmap"):
        entity_type = st.selectbox("Select Entity Type", options=['organizations', 'people', 'projects', 'topics'],
                                   format_func=lambda x: x.capitalize())
        with st.spinner(f"Generating co-occurrence map for {entity_type}..."):
            heatmap_fig = create_cooccurrence_heatmap(df, entity_type=entity_type)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)


def investigation_files_page():
    st.title("🗂️ Investigation Files: Original Narratives")
    st.markdown("Explore the original, machine-generated thematic stories.")
    narratives_dict = load_precomputed_pkl('narratives_dict.pkl')
    if narratives_dict is None:
        st.warning("Could not load 'narratives_dict.pkl'. This page requires pre-computed data.")
        return

    search_query = st.text_input("Search Narrative Titles...", placeholder="Type to filter...")
    filtered_titles = [t for t in list(narratives_dict.keys()) if
                       search_query.lower() in t.lower()] if search_query else list(narratives_dict.keys())

    if not filtered_titles:
        st.warning("No narratives match your search.")
        return

    selected_title = st.selectbox("Select a narrative to read:", options=filtered_titles)
    if 'processed_title' not in st.session_state or st.session_state.processed_title != selected_title:
        story_data = narratives_dict[selected_title]
        actors = story_data.get('actors', [])
        body = story_data.get('body', '')
        keywords = extract_keywords_from_title(selected_title)
        highlighted_story = highlight_text(body, actors, 'actor-highlight')
        st.session_state.highlighted_story_html = highlight_text(highlighted_story, keywords, 'keyword-highlight')
        st.session_state.current_actors = actors
        st.session_state.processed_title = selected_title

    col1, col2 = st.columns([1, 2.5])
    with col1:
        with st.container(border=True):
            st.markdown("##### Case File Details")
            st.markdown(f"**Title:** {st.session_state.processed_title}")
            st.markdown("---")
            st.markdown("**Primary Actors:**")
            if st.session_state.current_actors:
                for actor in st.session_state.current_actors: st.markdown(f"- {actor}")
            else:
                st.caption("None defined.")
    with col2:
        st.markdown(st.session_state.highlighted_story_html, unsafe_allow_html=True)


def entity_relationship_network_page():
    """The new network page, styled with a LIGHT theme."""
    st.title("🕸️ Entity Relationship Network")
    st.markdown(
        "Select a story to visualize the key entities and concepts discussed within it. This network is styled to match the investigation example.")

    stories = load_stories_from_json(THEMATIC_STORIES)
    if not stories:
        return

    all_titles = list(stories.keys())
    example_title = "New Member Added to Northern Border Partners' Audit Committee: Dan Dienstbier"
    default_index = all_titles.index(example_title) if example_title in all_titles else 0

    selected_title = st.selectbox("Select a story to analyze:", options=all_titles, index=default_index)

    if selected_title:
        selected_story = stories[selected_title]
        title = selected_story['title']
        summary = selected_story['summary']

        st.markdown("---")
        st.subheader(title)
        st.markdown(
            f"<div style='background-color:#F0F2F6; color:black; padding:15px; border: 1px solid #dee2e6; border-radius:5px;'>{summary}</div>",
            unsafe_allow_html=True)
        st.subheader("Entity Relationship Network")

        with st.spinner("Extracting entities and building network..."):
            example_entities = [
                'pipeline statements', 'SOCAL', 'invoice', 'deal', 'EOL logic issues', 'BP', 'Settlements',
                'third party gas', 'volumes allocation', 'anne.bike@enron.com', 'houston.ward@enron.com', 'Enron Corp',
                'imbalances', 'settlements', 'failed desk to desk deals', 'third party deals', 'P/L', 'volumes',
                'Redwood Gas', 'allocation', 'processes tightening', 'houston.ward@enron.com', 'anne.bike@enron.com',
                'volume management', 'back office groups', 'payments', 'meeting with settlements', 'Wood report',
                'Enron', 'SOCAL imbalances'
            ]

            graph_html = generate_styled_entity_network(title, summary, example_entities)

        if graph_html:
            components.html(graph_html, height=620)
        else:
            st.warning("Could not generate a network graph for this story.")


def timeline_page():
    st.title("🗓️ Event Timeline: Activity Spikes")
    st.markdown("This timeline plots periods of high email activity from the original dataset.")
    activity_spikes = load_precomputed_pkl('activity_spikes.pkl')
    activity_spike_map = load_precomputed_pkl('activity_spike_map.pkl')
    if not activity_spikes or not activity_spike_map:
        st.warning(
            "Could not load 'activity_spikes.pkl' or 'activity_spike_map.pkl'. This page requires pre-computed data.")
        return

    df_spikes = pd.DataFrame(activity_spikes)
    df_spikes['date'] = pd.to_datetime(df_spikes['date'])
    df_spikes['event_name'] = df_spikes['title'].apply(lambda x: x.split(': ')[-1])
    fig = px.scatter(df_spikes, x='date', y='email_count', size='email_count', color='event_name', hover_name='title',
                     title="Timeline of Key Activity Spikes")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.header("Investigate a Specific Event")
    selected_title = st.selectbox("Select an Event Spike:", options=list(activity_spike_map.keys()))
    if selected_title:
        spike_data = activity_spike_map[selected_title]
        summary_text = spike_data['summary']
        participants = spike_data.get('participants', [])
        orgs = spike_data.get('organizations', [])
        all_actors = [p.strip() for sublist in participants for p in sublist.split(',') if p.strip()] + orgs
        actor_counts = Counter(all_actors)

        col1, col2 = st.columns([1, 2])
        with col1:
            with st.container(border=True, height=600):
                st.markdown("##### Event Details")
                st.metric("Email Volume", f"{spike_data['email_count']:,}")
                st.metric("Duration", f"{spike_data.get('duration_days', 1)} Days")
                st.markdown("---")
                st.markdown("**Top Actors:**")
                if actor_counts:
                    for actor, count in sorted(actor_counts.items(), key=lambda i: i[1], reverse=True)[:15]:
                        st.caption(f"- {actor.split('@')[0]} ({count})")
        with col2:
            st.markdown(f"<div class='storybook-box'>{summary_text}</div>", unsafe_allow_html=True)

        st.subheader(f"Communication Network for '{selected_title}'")
        with st.spinner("Generating network..."):
            network_html = generate_narrative_network_graph(selected_title, list(actor_counts.keys()))
            if network_html: components.html(network_html, height=610)


# --- MAIN ROUTER ---
if page == "Dashboard":
    dashboard_page()
elif page == "AI Summary":
    ai_summary_page()
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Entity Relationship Network":
    entity_relationship_network_page()
elif page == "Event Timeline":
    timeline_page()