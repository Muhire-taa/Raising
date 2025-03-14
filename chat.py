import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

# Text Preprocessing Functions

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def process_feedback_data(df):
    feedback_entries = []
    for _, row in df.iterrows():
        if pd.notnull(row['most_recommend_rtv_program_reason']):
            feedback_entries.append({
                'program': row['most_recommend_rtv_program'],
                'feedback': row['most_recommend_rtv_program_reason'],
                'type': 'positive'
            })
        if pd.notnull(row['least_recommend_rtv_program_reason']):
            feedback_entries.append({
                'program': row['least_recommend_rtv_program'],
                'feedback': row['least_recommend_rtv_program_reason'],
                'type': 'negative'
            })
    return feedback_entries


# Embedding Generation & FAISS Indexing

def generate_feedback_embeddings(feedback_entries, model):
    texts = [clean_text(entry['feedback']) for entry in feedback_entries]
    embeddings = model.encode(texts)
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Retrieval System

def retrieve_feedback(query, model, index, feedback_entries, top_k=3):
    query_clean = clean_text(query)
    query_embedding = model.encode([query_clean]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved = []
    for idx in indices[0]:
        if idx < len(feedback_entries):
            retrieved.append(feedback_entries[idx])
    return retrieved


# Simulated Generative Response 

def generate_response(query, retrieved_feedback):
    prompt = f"Query: {query}\n\nRelevant community feedback:\n"
    for feedback in retrieved_feedback:
        prompt += f"- [{feedback['type']}] {feedback['program']}: {feedback['feedback']}\n"
    prompt += "\nBased on the above, provide a concise analysis."
    
    # Simulated generative model response
    response = (
        "Based on the feedback, Agriculture & Nutrition programs are highly valued for their support for local farmers, "
        "while WASH programs face challenges in maintenance. Stakeholders should consider these insights for targeted improvements."
    )
    return prompt, response


# Main Chatbot Interface 

def main():
    st.title("RTV Program Feedback Chatbot")
        
    # Load df
    if 'df' not in globals():
        st.warning("DataFrame 'df' not found. Using sample data for demonstration.")
        sample_data = [
            {
                'most_recommend_rtv_program': 'Agriculture & Nutrition',
                'least_recommend_rtv_program': 'WASH',
                'most_recommend_rtv_program_reason': 'Provides excellent support and resources for local farmers.',
                'least_recommend_rtv_program_reason': 'Suffers from poor maintenance and inadequate facilities.'
            },
            {
                'most_recommend_rtv_program': 'Health Access',
                'least_recommend_rtv_program': 'Agriculture & Nutrition',
                'most_recommend_rtv_program_reason': 'Offers prompt medical services and affordable care.',
                'least_recommend_rtv_program_reason': 'Has inefficient management and resource allocation.'
            },
            {
                'most_recommend_rtv_program': 'WASH',
                'least_recommend_rtv_program': 'Health Access',
                'most_recommend_rtv_program_reason': 'Improves water quality and sanitation effectively.',
                'least_recommend_rtv_program_reason': 'Involves long waiting times and limited service coverage.'
            }
        ]
        df_sample = pd.DataFrame(sample_data)
        global df
        df = df_sample
    
    # Process the DataFrame to extract feedback entries
    feedback_entries = process_feedback_data(df)
    
    # Load the Sentence Transformer model
    with st.spinner("Loading embedding model..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings and build the FAISS index
    embeddings = generate_feedback_embeddings(feedback_entries, model)
    index = build_faiss_index(embeddings)
    
    st.write("Enter a query related to community feedback on RTV programs. Examples:")
    st.write("- What reasons do communities give for recommending Agriculture & Nutrition programs?")
    st.write("- What concerns are expressed about WASH programs?")
    st.write("- Show me feedback from communities that preferred Health Access programs.")
    
    query = st.text_input("Your Query:")
    
    if query:
        retrieved_feedback = retrieve_feedback(query, model, index, feedback_entries)
        prompt, chatbot_response = generate_response(query, retrieved_feedback)
        
        st.subheader("Constructed Prompt for Generative Model:")
        st.code(prompt, language="text")
        
        st.subheader("Chatbot Response:")
        st.write(chatbot_response)
        
        st.subheader("Retrieved Feedback Entries:")
        for fb in retrieved_feedback:
            st.write(f"[{fb['type'].capitalize()}] {fb['program']}: {fb['feedback']}")
    
if __name__ == '__main__':
    main()
