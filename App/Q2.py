

import json
import faiss
import numpy as np
import gradio as gr
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from langchain_core.documents import Document
from persona import Users  # Assuming Users is a list of user profiles
from together import Together
import os 
from dotenv import load_dotenv
import streamlit as st
## for loading environment variable say (API_KEY)
load_dotenv()
# Load embeddings model
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
# Initialize FAISS Index
dimension = embeddings.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
# Define embedding function
embedding_function = lambda text: embeddings.encode(text, normalize_embeddings=True).tolist()
# Create an in-memory docstore
docstore = InMemoryDocstore()
# Create vector store
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id={},
)
# Create persona-based documents
documents = []
for user in Users:
    name = user['Name']
    location = user['Location']
    Interest = user['Interests']
    BackGround = user['Background']
    views = user['Values & Beliefs']
    context = f"Location: {location}\nInterests: {Interest}\nBackground: {BackGround}\nValues & Beliefs: {views}"
    info = Document(page_content=context, metadata={'Name': name, 'Location': location, 'Background': BackGround})
    documents.append(info)
# Generate UUIDs for documents
uuids = [str(uuid4()) for _ in documents]
# Encode document embeddings
doc_vectors = np.array(
    [embeddings.encode(doc.page_content, normalize_embeddings=True) for doc in documents], 
    dtype=np.float32  # FAISS requires float32
)
# Add vectors to FAISS index
index.add(doc_vectors)  
# Store document mappings
docstore.add({uuids[i]: documents[i] for i in range(len(documents))}) 
# Update FAISS index-to-docstore mapping
vector_store.index_to_docstore_id = {i: uuids[i] for i in range(len(documents))}
# Query and get results
def get_similar_user(query):
    # Encode query to vector
    query_vector = np.array([embeddings.encode(query, normalize_embeddings=True)], dtype=np.float32)
    # Perform FAISS search
    k = 5  # Retrieve top 5 matches
    scores, indices = index.search(query_vector, k)
    # Set similarity threshold (75% = 0.75)
    threshold = 0.65  
    # Filter results based on similarity score
    filtered_results = [
        (docstore.search(uuids[idx]), scores[0][i]) 
        for i, idx in enumerate(indices[0]) if scores[0][i] >= threshold
    ]
    # Construct response string
    if filtered_results:
        result_str = "\nTop Matching Profiles (Similarity ≥ 75%):\n"
        for doc, score in filtered_results:
            # print(score)
            result_str += (
                f"\n- **Name:** {doc.metadata['Name']}\n"
                f"\n- **Location:** {doc.metadata['Location']}\n"
                
                f"  **Profile Summary:** {doc.page_content}\n"
            )
    else:
        result_str = "No matches found with similarity ≥ 75%."

    return result_str  # Returning as a formatted string

### this will use the retrieved similar user and then recommed based on the factors (background,interest,belief and so on)
def Recommend_user(query):
    client = Together(api_key=os.getenv("API_KEY"))
    system_prompt="""You are an AI-powered social matching assistant that helps users find like-minded individuals based on their background, interests, and values. Given a user's context and persona model, your goal is to return the best-fit people along with a connection potential/compatibility percentage and insights explaining why each person was recommended.

### **Response Format:**

#### **1. Top Recommendation:**  
**Name:** [Person's Name]  
**Location:** [Location of Person]  
**Connection Potential:** [Percentage (0-100%) reflecting compatibility]  

##### **Reason for Recommendation:**  
- **Shared Interests:** [Common interests between the user and the recommended person]  
- **Background Alignment:** [Relevant similarities in background, education, or profession]  
- **Values & Beliefs Match:** [Any shared beliefs, lifestyle choices, or principles]  
- **Additional Insights:** [Why this connection might be valuable or meaningful]  

#### **2. Second Match:**  
**Name:** [Person's Name]  
**Location:** [Location of Person]  
**Connection Potential:** [Percentage (0-100%)]  

##### **Reason for Recommendation:**  
- **Shared Interests:** [Common interests]  
- **Background Alignment:** [Relevant similarities]  
- **Values & Beliefs Match:** [Shared principles]  
- **Additional Insights:** [Why this connection is valuable]  

#### **3. Third Match:**  
**Name:** [Person's Name]  
**Location:** [Location of Person]  
**Connection Potential:** [Percentage (0-100%)]  

##### **Reason for Recommendation:**  
- **Shared Interests:** [Common interests]  
- **Background Alignment:** [Relevant similarities]  
- **Values & Beliefs Match:** [Shared principles]  
- **Additional Insights:** [Why this connection is valuable]  

### **If No Matches Are Found:**  
*"I'm sorry, but I don't have enough information to make a recommendation at this time."*  

**Do not add any extra information, explanations, or commentary outside of this format. Return only the requested details.**

"""
    context=get_similar_user(query)
    print(context)
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{"role": "system", "content":system_prompt},{"role": "user", "content": f"use this context {context}"}],
    )
    return response.choices[0].message.content



# query="I just moved to NYC and I’m looking for people who are passionate about AI/ML research, nature conservation, or hiking, and who share my values"
# print(Recommend_user(query))

def gradio_interface(user_input):
    if not user_input.strip():
        return "⚠️ Please enter a valid query to proceed."
    
    return Recommend_user(user_input)

# Launch Gradio app(built in interface alternative of )
gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Describe your interests and values"),
    outputs=gr.Markdown(label="Recommended Matches"),
    title="🔍 AI-Powered Social Matcher",
    description="Enter your preferences, and we'll find the best people for you!",
    theme="compact"
).launch()

# query="""Recommend a professional who is an entrepreneur and expert in AI/ML, Neuromorphic Computing, and Sustainable Energy. The person should have a background in physics, astrophysics, and robotics, and be focused on AI-driven solutions for user engagement, ethical AI, and sustainability"""
# print(get_similar_user(query))