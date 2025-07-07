import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from collections import Counter

from src.config.config import PNG_IMAGE, CACHED_CLUSTER_STORIES, CACHED_STORIES_PATH, THEMATIC_STORIES

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
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


# --- Data Loading and Processing Functions (with Caching) ---
@st.cache_data
def load_conversation_data():
    with open(CACHED_CLUSTER_STORIES, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.rename(columns={'cluster_id': 'id', 'title': 'topic', 'summary': 'summary', 'email_count': 'email_count'},
              inplace=True)
    return df


@st.cache_data
def load_narrative_data():
    with open(THEMATIC_STORIES, 'r', encoding='utf-8') as f:
        content = f.read()
    stories_raw = re.split(r'\n📚 Theme-\d+\n============================================================\n', content)
    stories = {}
    for story_text in stories_raw:
        if "Title:" in story_text:
            try:
                title = re.search(r'Title: (.*?)\n', story_text).group(1).strip()
                body = story_text.split(title, 1)[1].strip()
                stories[title] = body
            except (AttributeError, IndexError):
                continue
    return stories


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


# --- Dynamic Entity and Keyword Extraction ---
@st.cache_data
def extract_proper_nouns(text):
    """Dynamically extracts capitalized words/phrases as a heuristic for actors."""
    pattern = r'\b(?:[A-Z][a-z\'-]+(?: |\'s )?)+(?:[A-Z][a-z\'-]+)*\b|\b[A-Z]{2,}\b'
    entities = re.findall(pattern, text)
    common_words_to_filter = {'The', 'This', 'In', 'A', 'An', 'On', 'It', 'For', 'With', 'As', 'From'}
    filtered_entities = {entity: count for entity, count in Counter(entities).items() if
                         entity not in common_words_to_filter and len(entity) > 2}
    return filtered_entities


def extract_keywords_from_title(title):
    stop_words = {'a', 'an', 'and', 'the', 'in', 'of', 'for', 'with', 'on', 'at', 'by', 'to', 'is', 'was', 'from',
                  'its'}
    words = re.split(r'\W+', title.lower())
    return [word for word in words if len(word) > 3 and word not in stop_words]


def highlight_text(text, terms_to_highlight, css_class):
    sorted_terms = sorted(list(set(terms_to_highlight)), key=len, reverse=True)
    for term in sorted_terms:
        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
        text = pattern.sub(f"<span class='{css_class}'>\\1</span>", text)
    return text


# --- Visualization and Graphing ---
@st.cache_data
def generate_network_graph(graph_id, text_content, central_node_name, actor_entities_dict):
    if not actor_entities_dict:
        return None

    # Define some known orgs and people for icon differentiation
    known_orgs_lower = {"enron", "dynegy", "ferc", "california", "andersen", "ees", "sec", "pg&e", "socal", "reliant",
                        "mirant", "chevron", "ubs", "citi", "morgan", "corp", "inc"}
    known_people_lower = {"lay", "skilling", "fastow", "watkins", "causey", "whalley", "davis"}

    net = Network(height='500px', width='100%', bgcolor='#FFFFFF', font_color='#333333', notebook=True,
                  cdn_resources='in_line')

    # **FIXED**: Using the correct physics solver
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=120)

    net.add_node(central_node_name, label=central_node_name, size=25, color='#FF5733', shape='star',
                 title=f"Central Topic: {central_node_name}")

    # **FIXED**: Iterate over dictionary items
    for entity, count in actor_entities_dict.items():
        size = 12 + count * 2
        entity_lower = entity.lower()

        if any(p in entity_lower for p in known_people_lower):
            shape, icon_code, color, title_prefix = 'icon', 'f007', '#0072B2', 'Person'
        elif any(o in entity_lower for o in known_orgs_lower):
            shape, icon_code, color, title_prefix = 'icon', 'f1ad', '#D55E00', 'Organization'
        else:
            shape, icon_code, color, title_prefix = 'dot', None, '#009E73', 'Entity'

        title = f"{title_prefix}: {entity} (Mentioned {count} times)"

        if shape == 'icon':
            net.add_node(entity, label=entity, title=title, size=size, shape=shape, color=color, fa_icon=icon_code,
                         fa_icon_color="#ffffff")
        else:
            net.add_node(entity, label=entity, title=title, size=size, shape=shape, color=color)

        net.add_edge(central_node_name, entity, value=count)

    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = os.path.join("temp", f"network_{re.sub(r'[^a-zA-Z0-9]', '_', str(graph_id))}.html")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(net.html)
    except Exception as e:
        st.error(f"Error saving graph: {e}")
        return None

    return file_path


# --- Load Data ---
conversation_df = load_conversation_data()
narratives_dict = load_narrative_data()
activity_spikes = load_activity_spike_data()
activity_spike_map = {spike['title']: spike for spike in activity_spikes}

# --- Sidebar ---
st.sidebar.image(PNG_IMAGE, use_column_width=True)
st.sidebar.title("The Enron Files")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select an Analysis View",
    ("Investigation Files", "Event Timeline"),
    captions=["Explore Narratives & Data", "Analyze Key Events"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This interactive application visualizes AI-generated summaries and stories from the Enron email corpus.")


# --- Page Implementations ---
def investigation_files_page():
    st.title("🗂️ Investigation Files")
    st.markdown("Explore the automatically generated narratives and the underlying conversation data.")
    tab1, tab2 = st.tabs(["📖 Generated Narratives", "📂 Conversation Dossiers"])

    with tab1:
        st.header("Narrative Explorer")

        # **NEW**: Search bar for narratives
        search_query_narrative = st.text_input("Search Narrative Titles...", placeholder="Type to filter narratives...")

        all_titles = list(narratives_dict.keys())
        if search_query_narrative:
            filtered_titles = [title for title in all_titles if search_query_narrative.lower() in title.lower()]
        else:
            filtered_titles = all_titles

        if not filtered_titles:
            st.warning("No narratives match your search term.")
            return

        selected_title = st.selectbox("Select a narrative file:", options=filtered_titles, index=0,
                                      label_visibility="collapsed")

        if selected_title:
            story_content = narratives_dict[selected_title]
            actors_dict = extract_proper_nouns(story_content)
            keywords = extract_keywords_from_title(selected_title)

            highlighted_story = highlight_text(story_content, list(actors_dict.keys()), 'actor-highlight')
            highlighted_story = highlight_text(highlighted_story, keywords, 'keyword-highlight')

            # **NEW**: Layout with columns for better differentiation
            col1, col2 = st.columns([1, 2.5])

            with col1:
                with st.container(border=True):
                    st.markdown("#### Case File Details")
                    st.markdown(f"**Title:** {selected_title}")
                    st.markdown("---")
                    st.markdown("**Identified Actors & Entities:**")
                    if actors_dict:
                        for actor, count in sorted(actors_dict.items(), key=lambda item: item[1], reverse=True)[:10]:
                            st.caption(f"- {actor} ({count} mentions)")
                    else:
                        st.caption("None identified.")

            with col2:
                st.markdown(f"<div class='storybook-box'>{highlighted_story}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Network of Actors in '{selected_title}'")
            with st.spinner("Generating network graph..."):
                # **FIXED**: Pass the actors_dict directly
                network_path = generate_network_graph(selected_title, story_content, selected_title, actors_dict)
                if network_path and os.path.exists(network_path):
                    with open(network_path, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=510)

    with tab2:
        st.header("Search Conversation Dossiers")
        search_query = st.text_input("Search topics...", label_visibility="collapsed",
                                     placeholder="Search for keywords (e.g., Dynegy, Merger, California)...")
        results = conversation_df[conversation_df.apply(
            lambda r: search_query.lower() in str(r['topic']).lower() or search_query.lower() in str(
                r['summary']).lower(), axis=1)] if search_query else conversation_df.sort_values('email_count',
                                                                                                 ascending=False).head(
            10)

        if not search_query:
            st.info("Showing the 10 most active conversations. Use the search bar for specific topics.")

        for _, row in results.iterrows():
            with st.expander(f"**{row['topic']}** ({row['email_count']} emails)"):
                st.markdown("**Summary:**")
                st.write(row['summary'])

                st.markdown("**Key Actor & Organization Network:**")
                dossier_text = row['topic'] + " " + row['summary']
                # **FIXED**: Create and pass the dictionary
                dossier_actors_dict = extract_proper_nouns(dossier_text)
                if dossier_actors_dict:
                    network_path_dossier = generate_network_graph(f"dossier_{row['id']}", dossier_text, row['topic'],
                                                                  dossier_actors_dict)
                    if network_path_dossier and os.path.exists(network_path_dossier):
                        with open(network_path_dossier, 'r', encoding='utf-8') as f:
                            components.html(f.read(), height=450)
                else:
                    st.caption("_No key actors identified to generate a network graph._")


def timeline_page():
    st.title("🗓️ Event Timeline Analysis")
    st.markdown(
        "This timeline plots high-activity periods in email traffic. Select an event to investigate the narrative and communication network.")

    df_spikes = pd.DataFrame(activity_spikes)
    if not df_spikes.empty:
        df_spikes['date'] = pd.to_datetime(df_spikes['date'])
        df_spikes['event_name'] = df_spikes['title'].apply(lambda x: x.split(': ')[-1])

        fig = px.scatter(df_spikes, x='date', y='email_count', size='email_count', color='event_name',
                         hover_name='event_name', hover_data={'date': True, 'email_count': True, 'duration_days': True},
                         title="Timeline of Key Activity Spikes", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Investigate a Specific Event")
        selected_spike_title = st.selectbox("Select an Event Spike:", options=list(activity_spike_map.keys()))

        if selected_spike_title:
            spike_data = activity_spike_map[selected_spike_title]

            # Structured actors for more accuracy in spikes
            participants_list = spike_data.get('participants', [])
            orgs_list = spike_data.get('organizations', [])
            actors_from_data = [p.strip() for sublist in participants_list for p in sublist.split(',') if
                                '@' in p] + orgs_list
            actor_counts = Counter(actors_from_data)

            # Highlight using the identified actors
            highlighted_summary = highlight_text(spike_data['summary'], list(actor_counts.keys()), 'actor-highlight')

            # **NEW**: Use column layout for clear differentiation
            col1, col2 = st.columns([1, 2.5])

            with col1:
                with st.container(border=True):
                    st.markdown("#### Event Details")
                    st.metric("Email Volume", f"{spike_data['email_count']:,}")
                    st.metric("Duration", f"{spike_data.get('duration_days', 1)} Days")
                    st.markdown("---")
                    st.markdown("**Key Actors & Entities:**")
                    if actor_counts:
                        for actor, count in sorted(actor_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
                            st.caption(f"- {actor.split('@')[0]} ({count} mentions)")
                    else:
                        st.caption("None identified.")

            with col2:
                st.markdown(f"<div class='storybook-box'>{highlighted_summary}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Communication Network for '{selected_spike_title}'")
            with st.spinner("Generating network graph..."):
                # **FIXED**: Pass the dictionary
                network_path = generate_network_graph(spike_data['title'], spike_data['summary'], spike_data['title'],
                                                      actor_counts)
                if network_path and os.path.exists(network_path):
                    with open(network_path, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=610)
                else:
                    st.info("No key actors found to generate a network graph for this event.")
    else:
        st.warning("No activity spike data found.")


# --- Main Page Router ---
if page == "Dashboard":
    st.warning("The Dashboard page has been disabled as per the new requirements. Please select another view.",
               icon="⚠️")
elif page == "Investigation Files":
    investigation_files_page()
elif page == "Event Timeline":
    timeline_page()