import streamlit as st  # Streamlit for web app UI
import pandas as pd  # DataFrame operations
import re  # Regular expressions
from pyvis.network import Network  # Network visualization
import streamlit.components.v1 as components  # For embedding HTML
import os  # OS operations
from collections import Counter  # Counting elements
import plotly.express as px  # Plotly for charts
import pickle  # For loading/saving Python objects
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
import json  # JSON file operations

from itertools import combinations  # For generating combinations

from src.app_helper.ai_summary_generator import generate_llm_summary
from src.config.config import ENRON_LOGO, PROCESSED_JSON_OUTPUT, CLUSTER_STORIES, THEMATIC_STORIES, PICKLE_DIR  # Config imports

# --- Configuration Setup ---
# The original config import is commented out to make this script self-contained.
# from src.config.config import PROCESSED_JSON_OUTPUT, ENRON_LOGO, PICKEL_FILE, THEMATIC_STORIES

# --- Paths are defined directly here. ---
# Ensure these files are in the same directory as your script or provide the correct path.
ENRON_LOGO = ENRON_LOGO  # Path to logo image
PICKLE_DIR = PICKLE_DIR  # Directory for .pkl files
PROCESSED_JSON_OUTPUT = PROCESSED_JSON_OUTPUT  # Main processed JSON file
CACHED_CLUSTER_STORIES = CLUSTER_STORIES  # Cluster stories JSON
THEMATIC_STORIES = THEMATIC_STORIES  # Thematic stories JSON


def call_app_helper():
    # Import your helper methods here, assuming it's located in app_helper.py
    from src.app_helper.pickel_generator import generate_data  # Import the data generation function

    # Call the helper function to generate required data
    generate_data()


# --- Advanced Analysis Imports ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Text vectorization
    from sklearn.decomposition import LatentDirichletAllocation  # Topic modeling
    from minisom import MiniSom  # Self-Organizing Map

    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False

# --- LangChain & Ollama Imports ---
try:
    from langchain_community.chat_models import ChatOllama  # Ollama LLM
    from langchain.schema import HumanMessage  # Message schema

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="The Enron Files: A Narrative Investigation",  # App title
    page_icon="📖",  # App icon
    layout="wide",  # Wide layout
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)


# --- Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# You can create a style.css file to customize further if needed
local_css("style.css")


# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_precomputed_pkl(file_name):
    path = os.path.join(PICKLE_DIR, file_name)
    if not os.path.exists(path):
        print(f"Pickle file {file_name} not found. Calling helper to regenerate data.")
        call_app_helper()  # Generate data if missing
        if os.path.exists(path):
            print(f"Pickle file {file_name} generated successfully.")
        else:
            raise FileNotFoundError(f"Failed to generate pickle file: {file_name}")

    # If file exists, load the pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


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


# --- LLM Function ---
# @st.cache_data
# def generate_llm_summary(file_path):
#     if not LANGCHAIN_AVAILABLE:
#         return "Could not generate summary: `langchain` libraries not installed. Please run `pip install langchain langchain-community`."
#
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             context = f.read()
#
#         llm = ChatOllama(model="mistral")
#
#         prompt = f"""
#         **Your Persona:** You are a master storyteller and corporate historian. Your expertise lies in synthesizing complex, fragmented information into a single, compelling, and historically accurate narrative.
#
#         **Your Task:** Analyze the provided context, which is a large text file containing over 100 individual stories, each summarizing a "theme" from the Enron email dataset. Your goal is to synthesize these disparate stories into a single, cohesive, and epic narrative chronicling the rise and fall of Enron.
#
#         **Important Context on the Input Data:** The provided text is not a single document. It is a compilation of over 100 thematically grouped summaries. These summaries may overlap, repeat information, and cover different time periods and actors. Your primary challenge is to connect the dots, de-duplicate information, and build one overarching chronological narrative from these fragments.
#
#         **Instructions for the Narrative:**
#         1.  **Identify the Overarching Timeline:** Piece together the events chronologically, from the early signs of strategic maneuvering in California and the Dynegy merger talks to the final collapse and its aftermath.
#         2.  **Weave a Cohesive Story:** Do not simply list facts or themes. Create a story with a clear **beginning** (the peak of Enron's power and early manipulative strategies), a **middle** (the desperate attempts to maintain the façade, like the Dynegy merger), and a **climax** (the rapid unraveling and bankruptcy).
#         3.  **Incorporate Key Themes:** Seamlessly integrate the recurring themes from the source text—such as the California energy crisis, the secretive Dynegy merger negotiations, manipulation of regulatory bodies (FERC, CPUC), deceptive accounting practices (hidden files, SPEs), and the internal culture of secrecy—into the narrative.
#         4.  **Tone:** Use a compelling, journalistic, and slightly dramatic tone to capture the scale of the corporate tragedy. Use phrases that evoke a sense of a "house of cards" or a "dance of deception."
#
#         **Required Output Format:**
#         You MUST structure your response using the exact following Markdown format and headings:
#
#         ---
#         ### **Title: [Create a compelling, dramatic title for the story]**
#
#         ### **Key Actors:**
#         *   **Enron Executives:** [List the key executives]
#         *   **Corporate Entities:** [List the key companies and partners]
#         *   **Regulatory & Government Bodies:** [List the key agencies and political figures]
#
#         ### **The Story**
#
#         #### **The Beginning: [Create a subtitle for the first phase of the story, e.g., A Web of Power and Profit]**
#         [Write the first part of the narrative here, covering Enron's peak and the initial signs of trouble.]
#
#         #### **The Middle: [Create a subtitle for the second phase, e.g., The Desperate Dance]**
#         [Write the middle part of the narrative, focusing on the escalating problems and major events like the Dynegy merger attempt.]
#
#         #### **The Climax: [Create a subtitle for the final phase, e.g., The House of Cards Collapses]**
#         [Write the climax of the story, detailing the company's rapid downfall, bankruptcy, and the reasons why.]
#
#         #### **The Conclusion: [Use a subtitle like Fallout and Legacy]**
#         [Conclude the story by summarizing the immediate fallout (bankruptcy, trials) and the long-term legacy of the scandal (e.g., Sarbanes-Oxley Act).]
#         ---
#
#         **CONTEXT TO ANALYZE:**
#         ---
#         {context}
#         ---
#         """
#
#         message = HumanMessage(content=prompt)
#         response = llm([message])
#         return response.content
#
#     except Exception as e:
#         return f"""
#         **Error: Could not connect to local LLM.**
#
#         Please ensure you have Ollama installed and running in the background.
#
#         **Setup Steps:**
#         1. Download and run Ollama from [ollama.com](https://ollama.com).
#         2. In your terminal, run the command: `ollama run mistral`
#         3. Make sure the Ollama application is running before you start this Streamlit app.
#
#         *Detailed Error: {e}*
#         """

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
    fig, ax = plt.subplots(figsize=(3, 3))  # Changed from (8, 8) to (6, 6)

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
if os.path.exists(ENRON_LOGO):
    st.sidebar.image(ENRON_LOGO, use_container_width=True)
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

    stories = load_stories_from_json(CLUSTER_STORIES)
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
        all_actors_for_counting = [p.strip() for sublist in participants for p in sublist.split(',') if
                                   p.strip()] + orgs
        actor_counts = Counter(all_actors_for_counting)

        # --- Start of New Code: Text Processing for Highlighting ---
        # 1. Define keywords to highlight
        # Get unique names and organizations from the data
        dynamic_keywords = list(set([p.split('@')[0].replace('.', ' ').title() for p in participants] + orgs))
        # Add other relevant domain terms
        domain_keywords = [
            'stock price', 'SEC probe', 'merger', 'acquisition', 'restructuring',
            'divestiture', 'conference calls', 'securities lawsuits', 'Ken Lay'
        ]
        # Combine, ensure uniqueness, and sort by length (longest first) to prevent partial matches
        # keywords_to_highlight = sorted(list(set(dynamic_keywords + domain_keywords)), key=len, reverse=True)

        # 2. Process the summary text
        # Separate the title from the body
        parts = summary_text.split('\n\n', 1)
        title_raw = parts[0].replace('**', '')
        body_raw = parts[1] if len(parts) > 1 else ""

        # Wrap title in its own HTML span with the 'summary-title' class
        title_html = f"<span class='summary-title'>{title_raw}</span>"

        # Iterate through keywords and wrap them in the 'highlighted-keyword' class
        # highlighted_body = body_raw
        # for keyword in keywords_to_highlight:
        #     # Use regex for case-insensitive, whole-word matching. re.escape handles special characters.
        #     pattern = re.compile(r'\b(' + re.escape(keyword) + r')\b', re.IGNORECASE)
        #     # The lambda function ensures that the original casing of the word is preserved
        #     replacement = lambda m: f"<span class='highlighted-keyword'>{m.group(1)}</span>"
        #     highlighted_body = pattern.sub(replacement, highlighted_body)
        #
        # # Combine the formatted title and body
        final_html_summary = f"{title_html}{body_raw}"
        # --- End of New Code ---

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
            # Display the fully formatted HTML summary
            st.markdown(f"<div class='storybook-box'>{final_html_summary}</div>", unsafe_allow_html=True)

        st.subheader(f"Communication Network for '{selected_title}'")
        with st.spinner("Generating network..."):
            # Use the original full actor list for the network graph
            all_actors_for_network = list(actor_counts.keys())
            network_html = generate_narrative_network_graph(selected_title, all_actors_for_network)
            if network_html:
                components.html(network_html, height=610)


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