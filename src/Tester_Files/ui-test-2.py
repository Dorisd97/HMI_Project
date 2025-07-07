import streamlit as st
import pandas as pd
import re
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from collections import Counter
import plotly.express as px
import pickle

# --- LangChain & Ollama Imports ---
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- App Configuration & Constants ---
# Import paths from your config file
from src.config.config import PNG_IMAGE, THEMATIC_STORIES, PICKEL_FILE

PICKLE_DIR = PICKEL_FILE

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


# --- DATA LOADING FUNCTION (FOR PKL FILES ONLY) ---
@st.cache_data
def load_precomputed_pkl(file_name):
    path = os.path.join(PICKLE_DIR, file_name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"Data file not found at '{path}'. Please run the `precompute_data.py` script. This page will not work.")
        return None


# --- LLM & VISUALIZATION FUNCTIONS ---
@st.cache_data
def generate_llm_summary(file_path):
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


@st.cache_data
def generate_global_narrative_network(narratives: dict) -> str | None:
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


# --- Sidebar ---
st.sidebar.image(PNG_IMAGE, use_container_width=True)
st.sidebar.title("The Enron Files")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select an Analysis View",
    ("AI Summary", "Investigation Files", "Network Visualizer", "Event Timeline"),
    captions=["AI Executive Summary", "Explore Original Narratives", "Visualize Narrative Networks",
              "Analyze Activity Spikes"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app uses pre-computed data for speed. Ensure Ollama is running for AI features.")


# --- PAGE IMPLEMENTATIONS ---
def ai_summary_page():
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


def investigation_files_page():
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

    # --- WIDGET CREATION ---
    # We get the user's current selection directly from the widget.
    selected_title = st.selectbox("Select a narrative to read:", options=filtered_titles)

    # --- "COMPARE AND PROCESS" LOGIC ---
    # We only re-run the expensive code if the user's selection has changed.
    # We use a new session_state key to track what we've already processed.
    if 'processed_title' not in st.session_state or st.session_state.processed_title != selected_title:
        story_data = narratives_dict[selected_title]
        actors = story_data.get('actors', [])
        body = story_data.get('body', '')
        keywords = extract_keywords_from_title(selected_title)

        # Perform expensive highlighting
        highlighted_story = highlight_text(body, actors, 'actor-highlight')
        st.session_state.highlighted_story_html = highlight_text(highlighted_story, keywords, 'keyword-highlight')

        # Store actors and the title we just processed
        st.session_state.current_actors = actors
        st.session_state.processed_title = selected_title

    # --- DISPLAY LOGIC ---
    # This part is always fast because it just reads from session state.
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
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Network Visualizer":
    network_visualizer_page()
elif page == "Event Timeline":
    timeline_page()