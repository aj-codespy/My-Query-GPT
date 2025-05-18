import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
# Import ChatGoogleGenerativeAI instead of GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA # Keep if needed for other purposes, but not used in the main generation chain now
from langchain.prompts import PromptTemplate # Keep if needed for other purposes
from langchain_community.document_loaders import PyPDFLoader # Keep if needed
from langchain.text_splitter import CharacterTextSplitter # Keep for vector store
import tempfile
import os
import google.api_core.exceptions # Import the specific exception

# ---- Page Setup ----
st.set_page_config(page_title="SQL RAG Generator", layout="wide")
st.title("📊 SQL Query Generator using Schema + RAG")

# Use a form to group inputs and add a submit button
with st.form("sql_generator_form"):
    # ---- Upload Inputs ----
    uploaded_pdf = st.file_uploader("📎 Upload Schema PDF", type=["pdf"])
    role_context = st.text_area("🧠 Role Context (describe the data, business logic, etc)", height=150)
    user_query = st.text_input("🔍 Ask your SQL query in natural language")

    # Add a submit button
    submit_button = st.form_submit_button("Generate SQL")


# ---- Gemini API Setup ----
# Using the provided API key directly as requested.
# NOTE: For production applications, it is highly recommended to use Streamlit secrets
# (st.secrets["GOOGLE_API_KEY"]) instead of hardcoding the API key for security.
GOOGLE_API_KEY = 'AIzaSyDtB4bETfNDyvpzA_NnBKMrr56rdiOE8bQ'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Helper Functions ----
# Use st.cache_resource to cache the vector store
# It will only re-run if the 'pdf_file' content changes (i.e., a new file is uploaded)
@st.cache_resource
def create_vector_store(pdf_file):
    # Read text from the uploaded PDF file
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Use LangChain's CharacterTextSplitter for vector store (compatible with FAISS/LangChain)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    # Embedding model remains the same
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def generate_sql_from_query(user_query, context_docs, role_context):
    # Use ChatGoogleGenerativeAI with gemini-1.5-flash
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        temperature=0,
        google_api_key=GOOGLE_API_KEY, # Use the hardcoded key
        max_tokens=None,
        timeout=30,
        max_retries=2
    )

    # Combine context documents into a single string
    context = "\n\n".join([doc.page_content for doc in context_docs if hasattr(doc, 'page_content')])

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant that writes SQL queries.

Given the following:
1. Role Context:
{{role_context}}

2. Schema Documentation:
{{schema_documentation}}

Write a syntactically correct SQL query that will return the desired result. Only output the SQL code without explanation.
"""),
        ("human", "{user_query}")
    ])

    # Create a simple chain
    chain = prompt | llm

    try:
        # Invoke the chain with the necessary inputs
        response = chain.invoke({
            "role_context": role_context,
            "schema_documentation": context,
            "user_query": user_query
        })
        # The response from ChatGoogleGenerativeAI is a ChatMessage object
        return response.content # Extract the string content
    except google.api_core.exceptions.NotFound as e:
        st.error(f"Error: The requested Gemini model was not found. Please check your Google API key, the model name ('gemini-1.5-flash'), and ensure the model is available in your region. Details: {e}")
        return None # Return None or an empty string to indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None


# ---- Main Processing ----
# Only process if the submit button is clicked and inputs are provided
if submit_button and uploaded_pdf and user_query and role_context:
    with st.spinner("Processing PDF and generating SQL..."):
        # Create vector store (cached)
        vector_store = create_vector_store(uploaded_pdf)

        # Retrieve relevant documents from the vector store
        retriever = vector_store.as_retriever(search_type="similarity", k=4)
        similar_docs = retriever.get_relevant_documents(user_query)

        # Generate SQL using vector store context and role context
        sql_output = generate_sql_from_query(user_query, similar_docs, role_context)

        # Only display output if sql_output is not None (i.e., API call was successful)
        if sql_output is not None:
            st.subheader("🧾 Generated SQL Query")
            st.code(sql_output, language="sql")
            st.download_button("📋 Copy SQL to Clipboard", data=sql_output, file_name="query.sql", mime="text/sql")

elif submit_button and (not uploaded_pdf or not user_query or not role_context):
    st.warning("⬆️ Please upload the schema PDF, add role context, and enter your query.")

