import sys
import os
# --- Add src to sys.path so we can import config ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import config

import streamlit as st
st.set_page_config(page_title="Email Entity Explorer", layout="wide")

import json
import spacy
from pathlib import Path
from typing import List, Dict, Any
import time

# Use the config path
cleaned_json_path = config.CLEANED_JSON_PATH
CONFIG_LOADED = True  # Since we handle the import above, this should always be True

@st.cache_resource
def load_nlp_model():
    return spacy.load('en_core_web_sm')

@st.cache_data(show_spinner="Loading emails...")
def load_emails_from_json(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            emails = json.load(f)
        return emails
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
        return []

def extract_entities_from_email(email: Dict[str, Any], nlp) -> Dict[str, Any]:
    subject = email.get('Subject', '')
    body = email.get('Body', '')
    combined_text = f"{subject}\n{body}"
    if not combined_text.strip():
        email_copy = email.copy()
        email_copy['Entities'] = []
        return email_copy
    doc = nlp(combined_text)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents]
    email_copy = email.copy()
    email_copy['Entities'] = entities
    return email_copy

def process_emails_batch(emails: List[Dict[str, Any]], nlp, batch_size: int = 50) -> List[Dict[str, Any]]:
    processed_emails = []
    total_emails = len(emails)
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(0, total_emails, batch_size):
        batch = emails[i:i + batch_size]
        for j, email in enumerate(batch):
            processed_email = extract_entities_from_email(email, nlp)
            processed_emails.append(processed_email)
            current_progress = (i + j + 1) / total_emails
            progress_bar.progress(current_progress)
            status_text.text(f"Processing email {i + j + 1} of {total_emails}")
    progress_bar.empty()
    status_text.empty()
    return processed_emails

def display_entity_stats(emails_with_entities: List[Dict[str, Any]]):
    entity_counts = {}
    total_entities = 0
    for email in emails_with_entities:
        entities = email.get('Entities', [])
        total_entities += len(entities)
        for entity in entities:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Emails", len(emails_with_entities))
    with col2:
        st.metric("Total Entities", total_entities)
    if entity_counts:
        st.subheader("Entity Types Found")
        entity_df = {"Entity Type": list(entity_counts.keys()), "Count": list(entity_counts.values())}
        st.bar_chart(entity_df, x="Entity Type", y="Count")

def main():
    st.title("📧 Enron Email Entity Explorer")
    # Config status
    if CONFIG_LOADED:
        st.success("✅ Config loaded successfully")
    else:
        st.warning("⚠️ Config not loaded, using fallback path")
    st.info(f"📁 Using file: `{cleaned_json_path}`")
    if not Path(cleaned_json_path).exists():
        st.error(f"❌ File not found: {cleaned_json_path}")
        st.write("Please ensure the cleaned JSON file exists at the specified path.")
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.write("**Current File:**")
    st.sidebar.code(Path(cleaned_json_path).name)
    file_size = Path(cleaned_json_path).stat().st_size / (1024 * 1024)
    st.sidebar.write(f"Size: {file_size:.2f} MB")
    batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=200, value=50, help="Number of emails to process at once")
    show_raw_data = st.sidebar.checkbox("Show raw email data", value=False)

    # NLP model
    try:
        nlp = load_nlp_model()
        st.sidebar.success("✅ spaCy model loaded")
    except OSError:
        st.error("❌ spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        return

    # Load and process emails
    if st.sidebar.button("🚀 Load & Process Emails"):
        emails = load_emails_from_json(cleaned_json_path)
        if not emails:
            st.error("❌ No emails loaded. Please check the file path and format.")
            return
        st.info(f"Loaded {len(emails)} emails. Starting entity extraction...")
        start_time = time.time()
        emails_with_entities = process_emails_batch(emails, nlp, batch_size)
        processing_time = time.time() - start_time
        st.session_state.emails_with_entities = emails_with_entities
        st.session_state.processing_time = processing_time
        st.success(f"✅ Processing complete! Took {processing_time:.2f} seconds")

    # Display results if available
    if 'emails_with_entities' in st.session_state:
        emails_with_entities = st.session_state.emails_with_entities
        st.header("📊 Entity Statistics")
        display_entity_stats(emails_with_entities)
        st.header("📋 Email Browser")
        col1, col2 = st.columns(2)
        with col1:
            all_entity_types = set()
            for email in emails_with_entities:
                for entity in email.get('Entities', []):
                    all_entity_types.add(entity['label'])
            entity_filter = st.multiselect("Filter by Entity Type", sorted(all_entity_types), default=[])
        with col2:
            search_term = st.text_input("Search in subjects")
        filtered_emails = emails_with_entities
        if entity_filter:
            filtered_emails = [email for email in filtered_emails if any(entity['label'] in entity_filter for entity in email.get('Entities', []))]
        if search_term:
            filtered_emails = [email for email in filtered_emails if search_term.lower() in email.get('Subject', '').lower()]
        st.write(f"Showing {len(filtered_emails)} of {len(emails_with_entities)} emails")
        if filtered_emails:
            subjects = [email.get('Subject', f"Email {i+1}") for i, email in enumerate(filtered_emails)]
            selected_idx = st.selectbox("Select an email:", range(len(subjects)), format_func=lambda i: f"{i+1}. {subjects[i][:100]}...")
            selected_email = filtered_emails[selected_idx]
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📄 Email Content")
                st.markdown(f"**Subject:** {selected_email.get('Subject', 'No Subject')}")
                if show_raw_data:
                    st.markdown("**Raw Email Data:**")
                    st.json(selected_email)
                body = selected_email.get('Body', '')
                entities = selected_email.get('Entities', [])
                if body and entities:
                    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
                    highlighted_body = body
                    for entity in sorted_entities:
                        start = entity.get('start', 0)
                        end = entity.get('end', 0)
                        if start < len(body) and end <= len(body):
                            entity_text = entity['text']
                            label = entity['label']
                            highlighted_text = f"**{entity_text}** `[{label}]`"
                            highlighted_body = highlighted_body[:start] + highlighted_text + highlighted_body[end:]
                    st.markdown("**Email Body:**")
                    st.markdown(highlighted_body)
                else:
                    st.markdown("**Email Body:**")
                    st.text(body)
            with col2:
                st.subheader("🏷️ Extracted Entities")
                entities = selected_email.get('Entities', [])
                if entities:
                    entities_by_type = {}
                    for ent in entities:
                        label = ent['label']
                        if label not in entities_by_type:
                            entities_by_type[label] = []
                        entities_by_type[label].append(ent['text'])
                    for label, texts in entities_by_type.items():
                        st.markdown(f"**{label}:**")
                        unique_texts = list(set(texts))
                        for text in unique_texts:
                            count = texts.count(text)
                            if count > 1:
                                st.write(f"• {text} ({count}x)")
                            else:
                                st.write(f"• {text}")
                        st.write("")
                else:
                    st.write("_No entities found_")
        else:
            st.write("No emails match the current filters.")

if __name__ == "__main__":
    main()
