import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# --- LangChain & Ollama Imports ---
try:
    from langchain_community.chat_models import ChatOllama  # Ollama LLM
    from langchain.schema import HumanMessage  # Message schema

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def create_pre_summary(full_context):
    """
    Step 1: Creates a condensed, factual summary from the large text file.
    This reduces the context size for the final narrative generation.
    """
    llm_fast = ChatOllama(model="mistral")

    pre_summary_prompt = f"""
    Analyze the following text, which contains over 100 thematically grouped stories from the Enron email dataset.
    Your task is to extract and condense the most critical, recurring information into a concise set of bullet points.
    Do NOT write a full story or narrative. Your output should only be a factual list.

    Organize the bullet points under these headings:
    - **Key Actors:** List the most frequently mentioned individuals, companies, and regulatory bodies.
    - **Chronological Events:** List the major events in the order they occurred (e.g., California Crisis, Dynegy Merger, Bankruptcy).
    - **Core Themes:** List the main recurring themes of manipulation (e.g., accounting fraud, regulatory influence, market manipulation, secret deals).

    This output will be used by another AI to write a final narrative. Be concise and factual.

    CONTEXT:
    ---
    {full_context}
    ---
    """

    try:
        # Creating the HumanMessage object with the prompt as content
        message = HumanMessage(content=pre_summary_prompt)

        # Invoke the model and pass the message
        response = llm_fast.invoke([message])  # Pass as a list of messages

        return response.content
    except Exception as e:
        raise e


def generate_final_narrative(condensed_context):
    """
    Step 2: Uses the pre-summary to generate the final, high-quality narrative.
    """
    llm_narrative = ChatOllama(model="mistral")

    final_prompt = f"""
    **Your Persona:** You are a master storyteller and corporate historian. Your expertise lies in synthesizing complex, fragmented information into a single, compelling, and historically accurate narrative.

    **Your Task:** Analyze the provided condensed summary of the Enron scandal. Your goal is to expand these key points into a single, cohesive, and epic narrative chronicling the rise and fall of Enron.

    **Instructions for the Narrative:**
    1.  **Weave a Cohesive Story:** Do not simply list facts. Create a story with a clear **beginning**, a **middle**, and a **climax**.
    2.  **Incorporate Key Themes:** Seamlessly integrate the themes from the summary into the narrative.
    3.  **Tone:** Use a compelling, journalistic, and slightly dramatic tone to capture the scale of the corporate tragedy.

    **Required Output Format:**
    You MUST structure your response using the exact following Markdown format and headings:
    ---
    ### **Title: [Create a compelling, dramatic title for the story]**

    ### **Key Actors:**
    *   **Enron Executives:** [List the key executives from the context]
    *   **Corporate Entities:** [List the key companies from the context]
    *   **Regulatory & Government Bodies:** [List the key agencies from the context]

    ### **The Story**
    #### **The Beginning: [Create a subtitle for the first phase]**
    [Write the first part of the narrative here.]
    #### **The Middle: [Create a subtitle for the second phase]**
    [Write the middle part of the narrative here.]
    #### **The Climax: [Create a subtitle for the final phase]**
    [Write the climax of the story here.]
    #### **The Conclusion: [Use a subtitle like Fallout and Legacy]**
    [Conclude the story here.]
    ---

    **CONDENSED SUMMARY TO ANALYZE:**
    ---
    {condensed_context}
    ---
    """

    try:
        # Creating the HumanMessage object with the final prompt as content
        message = HumanMessage(content=final_prompt)

        # Invoke the model and pass the message
        response = llm_narrative.invoke([message])  # Pass as a list of messages

        return response.content
    except Exception as e:
        raise e


def generate_llm_summary(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_context = f.read()

        # Step 1: Create the pre-summary
        condensed_context = create_pre_summary(full_context)

        # Step 2: Generate the final narrative from the pre-summary
        final_narrative = generate_final_narrative(condensed_context)

        return final_narrative

    except Exception as e:
        return f"Error: {e}"