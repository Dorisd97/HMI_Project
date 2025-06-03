# src/visualisation/clustering_app.py

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config.config import *
from src.analysis.clustering_and_stories import EmailClusteringAndStories


def load_existing_results():
    """Load existing clustering results if they exist"""
    try:
        # Load stories
        if os.path.exists(CLUSTER_STORIES_PATH):
            with open(CLUSTER_STORIES_PATH, 'r') as f:
                stories = json.load(f)
        else:
            stories = {}

        # Load summary
        if os.path.exists(CLUSTER_SUMMARY_PATH):
            with open(CLUSTER_SUMMARY_PATH, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        # Load emails with clusters
        if os.path.exists(EMAILS_WITH_CLUSTERS_PATH):
            df = pd.read_csv(EMAILS_WITH_CLUSTERS_PATH)
        else:
            df = pd.DataFrame()

        return stories, summary, df

    except Exception as e:
        st.error(f"Error loading existing results: {str(e)}")
        return {}, {}, pd.DataFrame()


def create_cluster_visualization(df, summary):
    """Create visualizations for cluster analysis"""
    if df.empty or 'cluster' not in df.columns:
        return None, None

    # Cluster size distribution
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_counts = cluster_counts[cluster_counts.index != -1]  # Exclude noise

    # Bar chart of cluster sizes
    fig_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster ID', 'y': 'Number of Emails'},
        title='Email Distribution Across Clusters'
    )
    fig_bar.update_layout(showlegend=False)

    # Pie chart for dense clusters
    if 'dense_clusters' in summary and summary['dense_clusters']:
        dense_cluster_data = {
            str(c): summary['cluster_sizes'].get(str(c), 0)
            for c in summary['dense_clusters']
        }

        fig_pie = px.pie(
            values=list(dense_cluster_data.values()),
            names=[f"Cluster {c}" for c in dense_cluster_data.keys()],
            title='Dense Clusters Distribution'
        )
    else:
        fig_pie = None

    return fig_bar, fig_pie


def main():
    st.set_page_config(
        page_title="Email Clustering & Story Generation",
        page_icon="📧",
        layout="wide"
    )

    st.title("📧 Email Clustering & Story Generation")
    st.markdown("Discover patterns in email data through automated clustering and story generation")

    # Sidebar for controls
    st.sidebar.header("🔧 Controls")

    # Load existing results
    stories, summary, df = load_existing_results()

    # Check if results exist
    results_exist = bool(stories and summary)

    if results_exist:
        st.sidebar.success(f"✅ Found existing results with {len(stories)} stories")
    else:
        st.sidebar.warning("⚠️ No existing results found")

    # Configuration section
    st.sidebar.subheader("Configuration")

    # Data source selection
    use_entities = st.sidebar.checkbox(
        "Use Entities JSON",
        value=False,
        help="Use extracted entities instead of refined emails"
    )

    # Clustering parameters
    st.sidebar.subheader("Clustering Parameters")
    eps = st.sidebar.slider("Clustering Sensitivity (eps)", 0.1, 1.0, CLUSTERING_CONFIG['eps'], 0.1)
    min_samples = st.sidebar.slider("Min Samples per Cluster", 2, 20, CLUSTERING_CONFIG['min_samples'])
    top_n = st.sidebar.slider("Number of Dense Clusters", 1, 15, CLUSTERING_CONFIG['top_n_clusters'])

    # Run clustering button
    if st.sidebar.button("🚀 Run Clustering Analysis", type="primary"):
        run_clustering_analysis(use_entities, eps, min_samples, top_n)
        st.rerun()

    # Main content area
    if results_exist:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📖 Stories", "📊 Analytics", "🗂️ Data"])

        with tab1:
            display_stories(stories)

        with tab2:
            display_analytics(summary, df)

        with tab3:
            display_data_tab(df, summary)
    else:
        # Show setup instructions
        st.info("👆 Use the sidebar to configure and run clustering analysis")

        # Show data file status
        st.subheader("📁 Data File Status")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Refined JSON:**")
            if os.path.exists(REFINED_JSON_PATH):
                st.success(f"✅ Found: {os.path.basename(REFINED_JSON_PATH)}")
            else:
                st.error(f"❌ Missing: {os.path.basename(REFINED_JSON_PATH)}")

        with col2:
            st.write("**Entities JSON:**")
            if os.path.exists(EXTRACTED_ENTITIES_JSON_PATH):
                st.success(f"✅ Found: {os.path.basename(EXTRACTED_ENTITIES_JSON_PATH)}")
            else:
                st.error(f"❌ Missing: {os.path.basename(EXTRACTED_ENTITIES_JSON_PATH)}")


def run_clustering_analysis(use_entities, eps, min_samples, top_n):
    """Run the clustering analysis with given parameters"""

    # Update config temporarily
    temp_config = CLUSTERING_CONFIG.copy()
    temp_config.update({
        'eps': eps,
        'min_samples': min_samples,
        'top_n_clusters': top_n
    })

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with st.spinner("Running clustering analysis..."):
            status_text.text("Initializing analyzer...")
            progress_bar.progress(10)

            # Initialize analyzer
            analyzer = EmailClusteringAndStories(use_entities=use_entities)
            analyzer.config = temp_config

            status_text.text("Loading data...")
            progress_bar.progress(25)
            analyzer.load_data()

            status_text.text("Performing clustering...")
            progress_bar.progress(50)
            analyzer.cluster_emails()

            status_text.text("Finding dense clusters...")
            progress_bar.progress(75)
            analyzer.find_dense_clusters()

            status_text.text("Generating stories...")
            progress_bar.progress(90)
            stories = analyzer.save_results()

            progress_bar.progress(100)
            status_text.text("Complete!")

            st.success(f"✅ Analysis complete! Generated {len(stories)} stories from dense clusters.")

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

    finally:
        progress_bar.empty()
        status_text.empty()


def display_stories(stories):
    """Display the generated stories"""
    st.header("📖 Generated Stories")

    if not stories:
        st.warning("No stories available. Run clustering analysis first.")
        return

    # Story selection
    cluster_ids = list(stories.keys())
    selected_cluster = st.selectbox(
        "Select a cluster to view its story:",
        cluster_ids,
        format_func=lambda x: f"Cluster {x}"
    )

    if selected_cluster:
        story_content = stories[selected_cluster]

        # Display story in a nice format
        st.markdown("---")
        st.markdown(story_content)

        # Download button for individual story
        st.download_button(
            label=f"📄 Download Cluster {selected_cluster} Story",
            data=story_content,
            file_name=f"cluster_{selected_cluster}_story.md",
            mime="text/markdown"
        )

    # Download all stories
    st.markdown("---")
    all_stories_text = "\n\n---\n\n".join([f"# Cluster {k}\n\n{v}" for k, v in stories.items()])
    st.download_button(
        label="📚 Download All Stories",
        data=all_stories_text,
        file_name="all_cluster_stories.md",
        mime="text/markdown"
    )


def display_analytics(summary, df):
    """Display analytics and visualizations"""
    st.header("📊 Cluster Analytics")

    if not summary:
        st.warning("No analytics available. Run clustering analysis first.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Clusters", summary.get('total_clusters', 0))

    with col2:
        st.metric("Dense Clusters", len(summary.get('dense_clusters', [])))

    with col3:
        st.metric("Noise Points", summary.get('noise_points', 0))

    with col4:
        total_emails = summary.get('noise_points', 0) + sum(summary.get('cluster_sizes', {}).values())
        st.metric("Total Emails", total_emails)

    # Visualizations
    if not df.empty:
        fig_bar, fig_pie = create_cluster_visualization(df, summary)

        col1, col2 = st.columns(2)

        with col1:
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

    # Configuration used
    if 'config_used' in summary:
        st.subheader("🔧 Configuration Used")
        config_df = pd.DataFrame.from_dict(summary['config_used'], orient='index', columns=['Value'])
        st.dataframe(config_df)


def display_data_tab(df, summary):
    """Display raw data and cluster information"""
    st.header("🗂️ Data Overview")

    if df.empty:
        st.warning("No data available. Run clustering analysis first.")
        return

    # Dataset info
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Emails", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        if 'cluster' in df.columns:
            unique_clusters = df['cluster'].nunique()
            st.metric("Unique Clusters", unique_clusters)

    # Show cluster distribution
    if 'cluster' in df.columns:
        st.subheader("Cluster Distribution")
        cluster_dist = df['cluster'].value_counts().sort_index()
        st.bar_chart(cluster_dist)

    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(100))

    # Download options
    st.subheader("📥 Download Data")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download Clustered Data (CSV)",
            csv_data,
            "clustered_emails.csv",
            "text/csv"
        )

    with col2:
        if summary:
            summary_json = json.dumps(summary, indent=2)
            st.download_button(
                "Download Summary (JSON)",
                summary_json,
                "cluster_summary.json",
                "application/json"
            )


if __name__ == "__main__":
    main()
