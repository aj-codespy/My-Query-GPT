import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# Corrected import for KnowledgeGraphIndex
from llama_index.core import KnowledgeGraphIndex
# Import LlamaIndex Document
from llama_index.core import Document as LlamaIndexDocument
import tempfile
import os

# ---- Page Setup ----
st.set_page_config(page_title="SQL RAG Generator", layout="wide")
st.title("📊 SQL Query Generator using Schema + RAG")

# ---- Upload Inputs ----
uploaded_pdf = st.file_uploader("📎 Upload Schema PDF", type=["pdf"])
role_context = st.text_area("🧠 Role Context (describe the data, business logic, etc)", height=150)
user_query = st.text_input("🔍 Ask your SQL query in natural language")

# ---- Gemini API Setup ----
if "GOOGLE_API_KEY" not in st.secrets:
    st.warning("Please add your Gemini API key in Streamlit secrets as `GOOGLE_API_KEY`")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- Helper Functions ----
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_vector_store(text):
    # Use LangChain's CharacterTextSplitter for vector store (compatible with FAISS/LangChain)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def create_graph_index(text):
    # Use LangChain's CharacterTextSplitter to split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    langchain_docs = splitter.create_documents([text])

    # Convert LangChain Documents to LlamaIndex Documents
    # LlamaIndex's from_documents expects its own Document type
    llama_index_docs = [LlamaIndexDocument(text=doc.page_content, metadata=doc.metadata) for doc in langchain_docs]

    llm = GoogleGenerativeAI(model="gemini-pro")
    # Use KnowledgeGraphIndex from llama_index.core with LlamaIndex Documents
    graph_index = KnowledgeGraphIndex.from_documents(llama_index_docs, llm=llm)
    return graph_index

def generate_sql_from_query(user_query, context_docs, role_context):
    llm = GoogleGenerativeAI(model="gemini-pro")

    # Combine context
    context = "\n\n".join([doc.page_content for doc in context_docs])
    full_prompt = f"""
You are a helpful assistant that writes SQL queries.

Given the following:
1. Role Context:
{role_context}

2. Schema Documentation:
{context}

3. User Query:
{user_query}

Write a syntactically correct SQL query that will return the desired result. Only output the SQL code without explanation.
"""

    return llm.invoke(full_prompt)

# ---- Main Processing ----
if uploaded_pdf and user_query and role_context:
    with st.spinner("Processing PDF and generating SQL..."):
        pdf_text = extract_pdf_text(uploaded_pdf)

        # Create both vector and graph indexes
        vector_store = create_vector_store(pdf_text)
        retriever = vector_store.as_retriever(search_type="similarity", k=4)
        similar_docs = retriever.get_relevant_documents(user_query)

        # Create graph index using LlamaIndex Documents
        graph_index = create_graph_index(pdf_text)

        # Getting a summary from the graph index might require a query engine
        # For now, we'll keep the placeholder or implement a simple query if needed.
        # Example of getting a simple response from the graph (might need tuning)
        try:
            graph_query_engine = graph_index.as_query_engine()
            graph_summary_response = graph_query_engine.query(f"Summarize key entities and relationships related to: {user_query}")
            graph_summary = str(graph_summary_response)
        except Exception as e:
             graph_summary = f"Could not generate graph insight: {e}"
             st.warning(f"Warning: Could not generate graph insight. Error: {e}")


        # Add graph knowledge summary into role_context
        extended_context = role_context + "\n\n" + "Graph Insight:\n" + graph_summary

        # Generate SQL
        sql_output = generate_sql_from_query(user_query, similar_docs, extended_context)

        st.subheader("🧾 Generated SQL Query")
        st.code(sql_output, language="sql")
        st.download_button("📋 Copy SQL to Clipboard", data=sql_output, file_name="query.sql", mime="text/sql")
else:
    st.info("⬆️ Upload the schema PDF, add role context, and enter your query to generate SQL.")
