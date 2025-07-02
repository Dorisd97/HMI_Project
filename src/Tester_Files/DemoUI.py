import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter
from src.config.config import GENERATED_THEME_STORY_PATH2

# Set page config
st.set_page_config(
    page_title="Enron Email Thematic Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File path - update this to your actual file path
JSON_FILE_PATH = GENERATED_THEME_STORY_PATH2


@st.cache_data
def load_data():
    """Load and parse the JSON data"""
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {JSON_FILE_PATH}")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
        st.stop()


def parse_date(date_str):
    """Parse date string to datetime object"""
    try:
        # Handle different date formats
        if '.' in date_str:
            return datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
        else:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def create_timeline_chart(emails_df):
    """Create timeline visualization"""
    emails_df['parsed_date'] = emails_df['date'].apply(parse_date)
    emails_df = emails_df.dropna(subset=['parsed_date'])
    emails_df['year_month'] = emails_df['parsed_date'].dt.to_period('M')

    timeline_data = emails_df.groupby(['year_month', 'classification']).size().reset_index(name='count')
    timeline_data['year_month_str'] = timeline_data['year_month'].astype(str)

    fig = px.bar(timeline_data,
                 x='year_month_str',
                 y='count',
                 color='classification',
                 title="Email Volume Over Time by Classification",
                 labels={'year_month_str': 'Month', 'count': 'Number of Emails'})

    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_tone_distribution(emails_df):
    """Create tone distribution chart"""
    tone_counts = emails_df['tone'].value_counts()

    fig = px.pie(values=tone_counts.values,
                 names=tone_counts.index,
                 title="Distribution of Email Tones")
    return fig


def create_entity_analysis(emails_df):
    """Analyze entities across emails"""
    all_orgs = []
    all_people = []
    all_locations = []

    for _, row in emails_df.iterrows():
        entities = row['entities']
        all_orgs.extend(entities.get('organizations', []))
        all_people.extend(entities.get('people', []))
        all_locations.extend(entities.get('locations', []))

    return {
        'organizations': Counter(all_orgs).most_common(10),
        'people': Counter(all_people).most_common(10),
        'locations': Counter(all_locations).most_common(10)
    }


def main():
    st.title("📧 Enron Email Thematic Analysis")
    st.markdown("---")

    # Load data
    data = load_data()

    # Sidebar with metadata
    st.sidebar.header("📊 Analysis Overview")
    metadata = data['analysis_metadata']
    st.sidebar.metric("Total Emails", metadata['total_emails'])
    st.sidebar.metric("Model Used", metadata['model_used'].title())
    st.sidebar.write(f"**Analysis Date:** {metadata['analysis_timestamp'][:10]}")

    # Create DataFrame from emails
    emails_df = pd.DataFrame(data['email_summaries'])

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎯 Themes Overview", "📈 Analytics", "📧 Email Explorer", "🌐 Entity Analysis", "📖 Narrative Timeline"])

    with tab1:
        st.header("Major Themes Identified")

        # Display themes analysis
        themes_text = data['themes_analysis']

        # Extract theme sections
        theme_sections = re.split(r'\d+\.\s+', themes_text)[1:]  # Split by numbered sections
        theme_titles = [
            "The Dynegy Merger Saga",
            "California Energy Crisis and Market Manipulation",
            "Financial Engineering and Special Purpose Entities",
            "The Unraveling and Bankruptcy",
            "Regulatory and Legal Battles",
            "Energy Trading Operations",
            "Corporate Culture and Internal Dynamics"
        ]

        for i, (title, content) in enumerate(zip(theme_titles, theme_sections)):
            with st.expander(f"**{i + 1}. {title}**", expanded=False):
                st.write(content.strip())

    with tab2:
        st.header("📈 Email Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Timeline chart
            timeline_fig = create_timeline_chart(emails_df)
            st.plotly_chart(timeline_fig, use_container_width=True)

        with col2:
            # Tone distribution
            tone_fig = create_tone_distribution(emails_df)
            st.plotly_chart(tone_fig, use_container_width=True)

        # Classification distribution
        st.subheader("Email Classifications")
        class_counts = emails_df['classification'].value_counts()
        class_fig = px.bar(x=class_counts.index, y=class_counts.values,
                           title="Distribution of Email Classifications")
        st.plotly_chart(class_fig, use_container_width=True)

    with tab3:
        st.header("📧 Email Explorer")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_classification = st.selectbox(
                "Filter by Classification",
                ["All"] + list(emails_df['classification'].unique())
            )

        with col2:
            selected_tone = st.selectbox(
                "Filter by Tone",
                ["All"] + list(emails_df['tone'].unique())
            )

        with col3:
            search_term = st.text_input("Search in summaries")

        # Apply filters
        filtered_df = emails_df.copy()
        if selected_classification != "All":
            filtered_df = filtered_df[filtered_df['classification'] == selected_classification]
        if selected_tone != "All":
            filtered_df = filtered_df[filtered_df['tone'] == selected_tone]
        if search_term:
            filtered_df = filtered_df[filtered_df['summary'].str.contains(search_term, case=False, na=False)]

        st.write(f"Showing {len(filtered_df)} of {len(emails_df)} emails")

        # Display emails
        for _, email in filtered_df.iterrows():
            with st.expander(f"📧 ID {email['id']}: {email['subject'][:50]}..." if len(
                    email['subject']) > 50 else f"📧 ID {email['id']}: {email['subject']}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Date:** {email['date']}")
                    st.write(f"**Subject:** {email['subject']}")
                    st.write(f"**Summary:** {email['summary']}")

                with col2:
                    st.write(f"**Classification:** {email['classification']}")
                    st.write(f"**Tone:** {email['tone']}")

                    # Show entities if available
                    entities = email['entities']
                    if entities.get('organizations'):
                        st.write(f"**Organizations:** {', '.join(entities['organizations'][:3])}")
                    if entities.get('people'):
                        st.write(f"**People:** {', '.join(entities['people'][:3])}")
                    if entities.get('locations'):
                        st.write(f"**Locations:** {', '.join(entities['locations'][:3])}")

    with tab4:
        st.header("🌐 Entity Analysis")

        entity_data = create_entity_analysis(emails_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Top Organizations")
            if entity_data['organizations']:
                org_df = pd.DataFrame(entity_data['organizations'], columns=['Organization', 'Count'])
                fig = px.bar(org_df, x='Count', y='Organization', orientation='h',
                             title="Most Mentioned Organizations")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top People")
            if entity_data['people']:
                people_df = pd.DataFrame(entity_data['people'], columns=['Person', 'Count'])
                fig = px.bar(people_df, x='Count', y='Person', orientation='h',
                             title="Most Mentioned People")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Top Locations")
            if entity_data['locations']:
                loc_df = pd.DataFrame(entity_data['locations'], columns=['Location', 'Count'])
                fig = px.bar(loc_df, x='Count', y='Location', orientation='h',
                             title="Most Mentioned Locations")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.header("📖 Narrative Timeline")

        timeline_text = data['narrative_timeline']

        # Parse timeline phases
        phases = re.split(r'\*\*\d+\.', timeline_text)[1:]  # Split by numbered phases
        phase_titles = [
            "Early Operations (normal business) [January 1992 - August 2000]",
            "Growing Problems (regulatory issues, investigations) [September 2000 - December 2001]",
            "Crisis Phase (mergers, financial troubles) [January 2002 - March 2002]",
            "Collapse (bankruptcy, legal consequences) [April 2002 - December 2006]"
        ]

        for i, (title, content) in enumerate(zip(phase_titles, phases)):
            st.subheader(f"Phase {i + 1}: {title}")

            # Clean up the content
            content = content.replace('**', '').strip()

            # Split into key emails and description
            sections = content.split('- Description:')
            if len(sections) >= 2:
                key_emails = sections[0].replace('- Key Emails:', '').strip()
                description = sections[1].split('- Connection to the broader narrative:')[0].strip()
                connection = sections[1].split('- Connection to the broader narrative:')[
                    1].strip() if '- Connection to the broader narrative:' in sections[1] else ""

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("**Key Emails:**")
                    st.write(key_emails)

                with col2:
                    st.write("**Description:**")
                    st.write(description)
                    if connection:
                        st.write("**Connection:**")
                        st.write(connection)
            else:
                st.write(content)

            st.markdown("---")


if __name__ == "__main__":
    main()
