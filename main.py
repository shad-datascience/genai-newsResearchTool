import os
import streamlit as st
import time
import asyncio
import os
import asyncio
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_classic.chains import RetrievalQAWithSourcesChain

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
asyncio.set_event_loop(asyncio.new_event_loop())

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.markdown('This app is created by: <a href="https://shad-datascience.guthub.io" target="_blank">Shad Jamil</a>', unsafe_allow_html=True)
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# This creates a "memory" for the app that persists across reruns.
if 'processed' not in st.session_state:
    st.session_state.processed = False

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}") # Added keys for stability
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
folder_path = "faiss_index"

# Initialize models once to avoid re-loading on every run

# 1. Using the ultra-fast Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7, 
    google_api_key=GOOGLE_API_KEY
)

# 2. Using the confirmed stable embedding model from your list
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=GOOGLE_API_KEY
)
if process_url_clicked:
    if not any(url.strip() for url in urls):
        st.sidebar.error("Please enter at least one URL.")
    else:
        with st.spinner("Processing URLs... This may take a moment."):
            # Load data
            st.text("Data Loading...Started...")
            loader = UnstructuredURLLoader(urls=[url for url in urls if url.strip()])
            data = loader.load()
            
            # Split data
            st.text("Text Splitting...Started...")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)
            
            # Create and save FAISS index
            st.text("Embedding Vector Building...Started...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(folder_path)
        
        st.success("URL processing complete! You can now ask a question below.")
        # --- Solution Part 2: Set the flag to True ---
        st.session_state.processed = True

# Only show the question input if processing is done.
if st.session_state.processed:
    query = st.text_input("Question: ")
    if query:
        if os.path.exists(folder_path):
            with st.spinner("Searching for answers..."):
                try:
                    vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                    
                    st.header("Answer")
                    st.write(result["answer"])

                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split("\n")
                        for source in sources_list:
                            st.write(source)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:

            st.error("Index not found. Please process URLs again.")


