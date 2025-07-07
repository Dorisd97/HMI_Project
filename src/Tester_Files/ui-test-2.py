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

from src.config.config import PROCESSED_JSON_OUTPUT, PNG_IMAGE, PICKEL_FILE, THEMATIC_STORIES

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

# --- App Configuration & Constants ---
# Assuming you have a config file like this
# For this example, paths are defined directly
PNG_IMAGE = PNG_IMAGE # Make sure this file exists
PICKLE_DIR = PICKEL_FILE # Make sure this directory exists
THEMATIC_STORIES = THEMATIC_STORIES # Make sure this file exists

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


local_css("style.css")


# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_precomputed_pkl(file_name):
    """Loads pre-computed data for the narrative/timeline pages."""
    path = os.path.join(PICKLE_DIR, file_name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Data file not found at '{path}'. The 'Investigation Files' and 'Event Timeline' pages may not work.")
        return None


@st.cache_data
def load_email_dataframe(filepath=PROCESSED_JSON_OUTPUT):
    """Loads the original email JSON for the new Dashboard page."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # if the JSON is wrapped in { "emails": [...] }, unwrap it;
        # if it’s already a list, just use it directly.
        if isinstance(data, dict) and 'emails' in data:
            emails = data['emails']
        elif isinstance(data, list):
            emails = data
        else:
            st.error(f"Unexpected JSON structure in '{filepath}'")
            return None

        df = pd.DataFrame(emails)
        df['date'] = pd.to_datetime(
            df['date'],
            format='%d.%m.%Y %H:%M:%S',
            errors='coerce'
        )
        df['full_text'] = df['subject'].fillna('') + ' ' + df['summary'].fillna('')
        return df

    except FileNotFoundError:
        st.error(f"The main data file '{filepath}' was not found. The 'Dashboard' page will not work.")
        return None
    except Exception as e:
        st.error(f"Error loading main data file: {e}")
        return None



# --- LLM & VISUALIZATION FUNCTIONS (Existing) ---
@st.cache_data
def generate_llm_summary(file_path):
    # ... (user's existing function, no changes needed)
    if not LANGCHAIN_AVAILABLE:
        return "Could not generate summary: `langchain` libraries not installed. Please run `pip install langchain langchain-community`."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            context = f.read()
        llm = ChatOllama(model="mistral")
        prompt = f"""
        As a financial investigator analyzing the Enron scandal, your task is to provide a high-level executive summary based on the provided text, which is a collection of automatically generated stories from Enron's emails.
        Your summary should:
        1.  Identify the main recurring themes (e.g., financial manipulation, regulatory issues, the Dynegy merger).
        2.  Name the key companies and individuals who appear frequently.
        3.  Describe the general timeline or progression of events as depicted in the stories.
        4.  Conclude with a synthesis of the key events that led to Enron's downfall, according to this specific text.
        Keep the summary concise and professional, suitable for a case overview.
        CONTEXT:\n---\n{context}\n---
        """
        message = HumanMessage(content=prompt)
        response = llm([message])
        return response.content
    except Exception as e:
        return f"""
        **Error: Could not connect to local LLM.**
        Please ensure you have Ollama installed and running in the background.
        **Setup Steps:**
        1. Download and run Ollama from [ollama.com](https://ollama.com).
        2. In your terminal, run the command: `ollama run mistral`
        *Detailed Error: {e}*
        """


def extract_keywords_from_title(title):
    # ... (user's existing function, no changes needed)
    stop_words = {'a', 'an', 'and', 'the', 'in', 'of', 'for', 'with', 'on', 'at', 'by', 'to', 'is', 'was', 'from',
                  'its'}
    words = re.split(r'\W+', title.lower())
    return [word for word in words if len(word) > 3 and word not in stop_words]


def highlight_text(text, terms_to_highlight, css_class):
    # ... (user's existing function, no changes needed)
    sorted_terms = sorted(list(set(terms_to_highlight)), key=len, reverse=True)
    for term in sorted_terms:
        clean_term = re.sub(r'\s*\(.*\)', '', term).strip()
        if not clean_term: continue
        pattern = re.compile(r'\b(' + re.escape(clean_term) + r')\b', re.IGNORECASE)
        text = pattern.sub(f"<span class='{css_class}'>\\1</span>", text)
    return text


@st.cache_data
def generate_narrative_network_graph(central_node_name: str, actors_list: list[str]) -> str | None:
    # ... (user's existing function, no changes needed)
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


@st.cache_data
def generate_global_narrative_network(narratives: dict) -> str | None:
    # ... (user's existing function, no changes needed)
    if not narratives: return None
    net = Network(height='700px', width='100%', bgcolor='#FFFFFF', font_color='#333333', notebook=False,
                  cdn_resources='in_line')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=200)
    seen_actors = set()
    for theme, data in narratives.items():
        net.add_node(theme, label=theme, shape='star', size=25, color='#FF5733', title=theme)
        for actor in data.get('actors', []):
            if actor not in seen_actors:
                net.add_node(actor, label=actor, shape='dot', size=15, color='#009E73', title=actor)
                seen_actors.add(actor)
            net.add_edge(theme, actor)
    return net.generate_html()


# --- NEW VISUALIZATION FUNCTIONS FOR DASHBOARD ---
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
    fig, ax = plt.subplots(figsize=(10, 10))
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


@st.cache_data
def create_cooccurrence_heatmap(df, entity_type='organizations', top_n=15):
    all_entities = df['entities'].apply(lambda x: x.get(entity_type, [])).explode().dropna()
    top_entities = all_entities.value_counts().nlargest(top_n).index.tolist()
    co_matrix = pd.DataFrame(0, index=top_entities, columns=top_entities)
    for _, row in df.iterrows():
        row_entities = list(set(e for e in row['entities'].get(entity_type, []) if e in top_entities))
        for e1, e2 in combinations(row_entities, 2):
            co_matrix.loc[e1, e2] += 1
            co_matrix.loc[e2, e1] += 1
    fig = px.imshow(co_matrix, text_auto=True, aspect="auto",
                    labels=dict(x="Entity", y="Entity", color="Co-occurrence Count"),
                    title=f'Co-occurrence of Top {top_n} {entity_type.capitalize()}')
    return fig


# --- Sidebar ---
st.sidebar.image(PNG_IMAGE, use_container_width=True, caption="The Enron Files")
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select an Analysis View",
    ("Dashboard", "AI Summary", "Investigation Files", "Event Timeline", "Network Visualizer"),
    captions=["High-level Data Overview", "AI Executive Summary", "Explore Original Narratives", "Analyze Activity Spikes", "Visualize Narrative Networks"]
)

st.sidebar.markdown("---")
st.sidebar.info("This app uses a mix of pre-computed data and live analysis. Ensure Ollama is running for AI features.")


# --- PAGE IMPLEMENTATIONS ---
def ai_summary_page():
    # ... (user's existing function, no changes needed)
    st.title("🤖 AI Summary")
    st.markdown("A high-level overview of the case, featuring an AI-generated summary from the thematic stories file.")
    st.header("AI-Generated Executive Summary")
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
    """NEW PAGE: Displays high-level stats and advanced research tools."""
    st.title("📊 Research Dashboard")
    st.markdown(
        "A deep-dive analysis of the raw email data, featuring high-level statistics and machine learning-driven insights.")

    df = load_email_dataframe()
    if df is None: return

    # --- High-Level Overview ---
    st.header("High-Level Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails Analyzed", len(df))
    col2.metric("Unique Senders", df['from'].nunique())
    col3.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

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

    # --- Advanced Analysis Section ---
    st.header("Advanced Research Tools")

    # --- Kohonen Map (SOM) ---
    with st.expander("🧠 Kohonen Map (SOM) Document Clustering"):
        st.info(
            """
            A Self-Organizing Map (SOM) is a neural network that maps high-dimensional data (like email text) onto a 2D grid.
            - **Darker areas** represent natural clusters (documents in these areas are textually similar).
            - **Lighter areas** are boundaries between clusters.
            """
        )
        if st.button("Train SOM and Generate Map"):
            with st.spinner("Training SOM... This is computationally intensive and may take some time."):
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
                    st.write(f"**Subjects for emails in cluster ({win_coords[0]}, {win_coords[1]}):**")
                    for idx in doc_indices[:10]: st.caption(f"- `{df.iloc[idx]['subject']}`")
                    if len(doc_indices) > 10: st.caption("...and more.")
            else:
                st.warning("Not enough data to generate a SOM. You may need a larger dataset.")

    # --- Topic Modeling ---
    with st.expander("📚 Latent Dirichlet Allocation (LDA) Topic Modeling"):
        st.markdown("LDA is an algorithm that discovers abstract 'topics' that occur in a collection of documents.")
        num_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=5, step=1, key="lda_slider")
        with st.spinner("Finding topics..."):
            discovered_topics = perform_lda(df['full_text'], n_topics=num_topics)
            st.subheader(f"Discovered Topics (Top 10 words each)")
            for topic in discovered_topics:
                st.markdown(topic)

    # --- Co-occurrence Heatmap ---
    with st.expander("🔗 Entity Co-occurrence Heatmap"):
        st.markdown(
            "This heatmap shows how often different named entities appear together in the same email, revealing potential relationships.")
        entity_type = st.selectbox("Select Entity Type", options=['organizations', 'people', 'projects', 'topics'],
                                   format_func=lambda x: x.capitalize())
        with st.spinner(f"Generating co-occurrence map for {entity_type}..."):
            heatmap_fig = create_cooccurrence_heatmap(df, entity_type=entity_type)
            st.plotly_chart(heatmap_fig, use_container_width=True)


def investigation_files_page():
    # ... (user's existing function, no changes needed)
    st.title("🗂️ Investigation Files: Original Narratives")
    st.markdown(
        "Explore the original, machine-generated thematic stories. This page is optimized for fast reading and searching.")
    narratives_dict = load_precomputed_pkl('narratives_dict.pkl')
    if narratives_dict is None: return
    st.header("Narrative Explorer")
    search_query = st.text_input("Search Narrative Titles...", placeholder="Type to filter...")
    all_titles = list(narratives_dict.keys())
    filtered_titles = [t for t in all_titles if search_query.lower() in t.lower()] if search_query else all_titles
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
    if 'highlighted_story_html' in st.session_state:
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


def network_visualizer_page():
    # ... (user's existing function, no changes needed)
    st.title("🕸️ Network Visualizer")
    st.markdown("Visualize the relationships between themes and actors from the original thematic stories.")
    narratives_dict = load_precomputed_pkl('narratives_dict.pkl')
    if narratives_dict is None: return
    st.header("Narrative Network Explorer")
    search_query = st.text_input("Search Narrative Titles to Visualize...", placeholder="Type to filter...")
    filtered_titles = [t for t in list(narratives_dict.keys()) if
                       search_query.lower() in t.lower()] if search_query else list(narratives_dict.keys())
    if not filtered_titles:
        st.warning("No narratives match your search.")
        return
    selected_title = st.selectbox("Select a narrative to visualize:", options=filtered_titles)
    st.subheader(f"Communication Network for '{selected_title}'")
    story_data = narratives_dict[selected_title]
    actors = story_data.get('actors', [])
    with st.spinner("Generating local network graph..."):
        graph_html = generate_narrative_network_graph(selected_title, actors)
        if graph_html:
            components.html(graph_html, height=610)
        else:
            st.info("No actors available to generate a network for this narrative.")
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.header("Global Theme-Actor Network")
    with st.expander("Click to generate the global network of all narratives (may be slow on first load)"):
        with st.spinner("Building full theme–actor graph..."):
            if 'global_narrative_graph' not in st.session_state:
                st.session_state.global_narrative_graph = generate_global_narrative_network(narratives_dict)
            if 'global_narrative_graph' in st.session_state and st.session_state.global_narrative_graph:
                components.html(st.session_state.global_narrative_graph, height=710)


def timeline_page():
    # ... (user's existing function, no changes needed)
    st.title("🗓️ Event Timeline: Activity Spikes")
    st.markdown("This timeline plots periods of high email activity from the original dataset.")
    activity_spikes = load_precomputed_pkl('activity_spikes.pkl')
    activity_spike_map = load_precomputed_pkl('activity_spike_map.pkl')
    if not activity_spikes or not activity_spike_map: return
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
        highlighted_summary = highlight_text(summary_text, list(actor_counts.keys()), 'actor-highlight')
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
            st.markdown(f"<div class='storybook-box'>{highlighted_summary}</div>", unsafe_allow_html=True)
        st.subheader(f"Communication Network for '{selected_title}'")
        with st.spinner("Generating network..."):
            network_html = generate_narrative_network_graph(selected_title, list(actor_counts.keys()))
            if network_html: components.html(network_html, height=610)


# --- MAIN ROUTER ---
if page == "AI Summary":
    ai_summary_page()
if page == "Dashboard":
    dashboard_page()
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Network Visualizer":
    network_visualizer_page()
elif page == "Event Timeline":
    timeline_page()