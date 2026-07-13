import os
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
DB_PATH = ".chroma_db"

st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖", layout="centered")

# Visual layout adjustments with colorful minimal design and blurred background
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        
        .stApp {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 50%, rgba(240, 147, 251, 0.3) 100%);
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
        }
        
        /* Blurred background image overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('/static/alfa.png');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            filter: blur(8px);
            opacity: 0.6;
            z-index: -1;
        }
        
        .block-container {
            padding-top: 3rem;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
            margin: 1rem;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        footer { visibility: hidden; }
        
        /* Colorful chat message styling */
        .stChatMessage {
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stChatMessage[data-testid="stChatMessage-assistant"] {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .stChatMessage[data-testid="stChatMessage-user"] {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        
        /* Colorful title styling */
        h1 {
            color: #667eea;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.2);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Colorful caption styling */
        .stCaption {
            color: #764ba2;
            font-weight: 500;
        }
        
        /* Colorful input styling */
        .stTextInput > div > div > input {
            border: 2px solid #667eea;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.95);
            color: #333;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #764ba2;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Colorful button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_backend():
    """Establishes global application instances for vector lookup and the model."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY environment credentials.")
        return None, None
    
    os.environ["GOOGLE_API_KEY"] = api_key
    
    if not os.path.exists(DB_PATH):
        st.error("Persistent vector directory not found. Please run ingest.py first.")
        return None, None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Context window width expanded to k=6 to handle cross-sentence context logic better
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    return retriever, llm

retriever, llm = initialize_rag_backend()

# Application state instantiation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I've loaded your document index. What would you like to know?"}
    ]

st.title("📄 Document Knowledge Base")
st.caption("Retrieval-Augmented Generation Chatbot powered by Gemini")

# Conversation rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Main execution frame loop
if user_query := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    if retriever and llm:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Searching document context..."):
                try:
                    # Isolate text payloads from matching chunk documents
                    matched_docs = retriever.invoke(user_query)
                    context_text = "\n\n".join([doc.page_content for doc in matched_docs])
                    
                    # RAG Guardrail Prompt Architecture
                    prompt_template = (
                        f"You are a precise document analysis assistant.\n"
                        f"Answer the question based strictly on the provided context. If the answer isn't present, "
                        f"say you don't know.\n\n"
                        f"--- CONTEXT ---\n{context_text}\n---------------\n\n"
                        f"Question: {user_query}\n"
                        f"Answer:"
                    )
                    
                    execution_result = llm.invoke(prompt_template)
                    output_text = execution_result.content
                    
                    response_placeholder.write(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                except Exception as e:
                    error_msg = f"An execution error occurred: {str(e)}"
                    response_placeholder.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.warning("Backend pipeline initialization failed. Check your environment keys.")