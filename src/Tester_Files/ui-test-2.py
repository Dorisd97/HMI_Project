import streamlit as st
import pandas as pd
import json
import re
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from collections import Counter
import plotly.express as px

from src.config.config import THEMATIC_STORIES, CACHED_CLUSTER_STORIES, CACHED_STORIES_PATH, PNG_IMAGE

# --- LangChain & Ollama Imports for LLM ---
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


local_css("style.css")


# --- Data Loading and Advanced Parsing Logic ---
@st.cache_data
def load_conversation_data():
    with open(CACHED_CLUSTER_STORIES, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.rename(columns={'cluster_id': 'id', 'title': 'topic', 'summary': 'summary', 'email_count': 'email_count'},
              inplace=True)
    return df


@st.cache_data
def load_structured_narrative_data():
    """Parses thematic stories, extracting Title, Actors, and Story Body into a structured dict."""
    with open(THEMATIC_STORIES, 'r', encoding='utf-8') as f:
        content = f.read()

    stories_raw = re.split(r'\n📚 Theme-\d+\n============================================================\n', content)
    stories_structured = {}
    for story_text in stories_raw:
        title_match = re.search(r'Title: (.*?)\n', story_text)
        if not title_match:
            continue

        title = title_match.group(1).strip()

        actors_section_match = (
                re.search(r'Actors\s*:(.*?)(?:Story:|Summary:|The story begins)',
                          story_text, re.DOTALL | re.IGNORECASE)
                or re.search(r'key players like (.*?)(?:[.,])', story_text, re.IGNORECASE)
        )
        if actors_section_match:
            raw = actors_section_match.group(1)
            # split on commas or “and”
            actors = re.split(r',\s*|\s+and\s+', raw)
        story_section_match = re.search(r'(Story:|Summary:|The story begins)(.*)', story_text,
                                        re.DOTALL | re.IGNORECASE)

        actors = []
        if actors_section_match:
            actor_lines = actors_section_match.group(1).strip().split('\n')
            actors = [actor.strip().lstrip('- ').strip() for actor in actor_lines if actor.strip()]

        body = story_section_match.group(2).strip() if story_section_match else title

        stories_structured[title] = {'actors': actors, 'body': body}

    return stories_structured


@st.cache_data
def load_activity_spike_data():
    with open(CACHED_STORIES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    activity_data = [item for item in data if item.get('type') == 'activity_burst']
    for item in activity_data:
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', item['title'])
        item['date'] = date_match.group(0) if date_match else '2000-01-01'
    activity_data.sort(key=lambda x: x['date'])
    return activity_data


# --- LLM Function ---
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

        CONTEXT:
        ---
        {context}
        ---
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
        3. Make sure the Ollama application is running before you start this Streamlit app.

        *Detailed Error: {e}*
        """


# --- Dynamic Highlighting and Entity Extraction ---
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


# --- Visualization ---
@st.cache_data
@st.cache_data
def generate_network_graph(central_node_name: str, actors_list: list[str]) -> str | None:
    if not actors_list:
        return None

    net = Network(
        height='500px',
        width='100%',
        bgcolor='#FFFFFF',
        font_color='#333333',
        notebook=False,
        cdn_resources='in_line'
    )
    # central theme
    net.add_node(central_node_name, label=central_node_name, shape='star', size=30)
    # one node per actor, plus an edge back to the theme
    for actor in actors_list:
        net.add_node(actor, label=actor, shape='dot', size=15)
        net.add_edge(central_node_name, actor)

    # optional: tweak physics to taste
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -6000,
          "springLength": 200
        }
      }
    }
    """)

    # return the full HTML string for embedding
    return net.generate_html()

@st.cache_data
def generate_global_network_html(narratives: dict) -> str | None:
    if not narratives:
        return None

    net = Network(
        height='700px',
        width='100%',
        bgcolor='#FFFFFF',
        font_color='#333333',
        notebook=False,
        cdn_resources='in_line'
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=200)

    seen_actors = set()
    # Add themes and connect to actors
    for theme, data in narratives.items():
        net.add_node(
            theme,
            label=theme,
            shape='star',
            size=25,
            color='#FF5733',
            title=theme
        )
        for actor in data['actors']:
            if actor not in seen_actors:
                net.add_node(
                    actor,
                    label=actor,
                    shape='dot',
                    size=15,
                    color='#009E73',
                    title=actor
                )
                seen_actors.add(actor)
            net.add_edge(theme, actor)

    return net.generate_html()


# --- Load Data ---
conversation_df = load_conversation_data()
narratives_dict = load_structured_narrative_data()
activity_spikes = load_activity_spike_data()
activity_spike_map = {spike['title']: spike for spike in activity_spikes}

# --- Sidebar ---
st.sidebar.image(PNG_IMAGE, use_column_width=True)
st.sidebar.title("The Enron Files")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select an Analysis View",
    ("Dashboard", "Investigation Files", "Event Timeline"),
    captions=["AI Summary & Overview", "Explore Narratives", "Analyze Key Events"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app uses a local AI model (Mistral) for analysis. Ensure Ollama is running.")


# --- Page Implementations ---
def dashboard_page():
    st.title("📊 Dashboard: AI-Powered Investigation")
    st.markdown(
        "A high-level overview of the case, featuring an AI-generated summary of the key events and actors from the narrative files.")

    st.header("AI-Generated Executive Summary")
    with st.spinner("Analyzing narratives with local AI model... This may take a moment on first run."):
        summary = generate_llm_summary(THEMATIC_STORIES)

    st.markdown(f"<div class='llm-summary-box'>{summary}</div>", unsafe_allow_html=True)


def investigation_files_page():
    st.title("🗂️ Investigation Files")
    st.markdown("Explore the generated narratives and their underlying conversation data.")

    # Narrative Section
    st.header("Narrative Explorer")
    search_query_narrative = st.text_input("Search Narrative Titles...", placeholder="Type to filter narratives...")

    all_titles = list(narratives_dict.keys())
    filtered_titles = [t for t in all_titles if
                       search_query_narrative.lower() in t.lower()] if search_query_narrative else all_titles

    if not filtered_titles:
        st.warning("No narratives match your search term.")
    else:
        selected_title = st.selectbox("Select a narrative file:", options=filtered_titles, index=0,
                                      label_visibility="collapsed")
        if selected_title and selected_title in narratives_dict:
            story_data = narratives_dict[selected_title]
            actors = story_data.get('actors', [])
            body = story_data.get('body', '')
            keywords = extract_keywords_from_title(selected_title)

            highlighted_story = highlight_text(body, actors, 'actor-highlight')
            highlighted_story = highlight_text(highlighted_story, keywords, 'keyword-highlight')

            col1, col2 = st.columns([1, 2.5])
            with col1:
                with st.container(border=True):
                    st.markdown("#### Case File Details")
                    st.markdown(f"**Title:** {selected_title}")
                    st.markdown("---")
                    st.markdown("**Primary Actors & Entities:**")
                    if actors:
                        for actor in actors:
                            st.markdown(f"- {actor}")
                    else:
                        st.caption("None defined in source.")
            with col2:
                st.markdown(f"<div class='storybook-box'>{highlighted_story}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Network of Actors in '{selected_title}'")
            with st.spinner("Generating network graph..."):
                graph_html = generate_network_graph(selected_title, actors)
                if graph_html:
                    components.html(graph_html, height=500, scrolling=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🌐 Global Theme–Actor Network")
            with st.spinner("Building full theme–actor graph…"):
                html = generate_global_network_html(narratives_dict)
            if html:
                components.html(html, height=700, scrolling=True)


def timeline_page():
    st.title("🗓️ Event Timeline Analysis")
    st.markdown(
        "This timeline plots periods of high email activity. Select an event to investigate its narrative and communication network.")

    df_spikes = pd.DataFrame(activity_spikes)
    if not df_spikes.empty:
        df_spikes['date'] = pd.to_datetime(df_spikes['date'])
        df_spikes['event_name'] = df_spikes['title'].apply(lambda x: x.split(': ')[-1])

        fig = px.scatter(df_spikes, x='date', y='email_count', size='email_count', color='event_name',
                         hover_name='title', hover_data={'date': True, 'email_count': True},
                         title="Timeline of Key Activity Spikes", template="plotly_white")
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

            col1, col2 = st.columns([1, 2.5])
            with col1:
                with st.container(border=True, height=600):
                    st.markdown("#### Event Details")
                    st.metric("Email Volume", f"{spike_data['email_count']:,}")
                    st.metric("Duration", f"{spike_data.get('duration_days', 1)} Days")
                    st.markdown("---")
                    st.markdown("**Top Actors & Entities:**")
                    if actor_counts:
                        for actor, count in sorted(actor_counts.items(), key=lambda i: i[1], reverse=True)[:15]:
                            st.caption(f"- {actor.split('@')[0]} ({count})")
                    else:
                        st.caption("None identified.")

            with col2:
                st.markdown(f"<div class='storybook-box'>{highlighted_summary}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Communication Network for '{selected_title}'")
            with st.spinner("Generating network graph..."):
                network_path = generate_network_graph(spike_data['title'], summary_text, spike_data['title'],
                                                      list(actor_counts.keys()))
                if network_path and os.path.exists(network_path):
                    with open(network_path, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=610)


# --- Main Page Router ---
if page == "Dashboard":
    dashboard_page()
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Event Timeline":
    timeline_page()