import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚", layout="wide")
st.title("📚 Chat with Your PDF Book")

# Initialize Embedding Model & LLM (cached so they don't reload on every click)
@st.cache_resource
def get_models():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        client_args={"http2": False}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        client_args={"http2": False}
    )
    return embeddings, llm

embedding_model, llm = get_models()

# Helper function to process and index an uploaded PDF
def process_pdf(uploaded_file, db_dir):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_dir
        )
        return vectorstore
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Sidebar: Upload Book ---
with st.sidebar:
    st.header("1. Upload Your Book")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Create a separate DB folder per uploaded book name so books don't mix
        safe_name = "".join(c if c.isalnum() else "_" for c in uploaded_file.name)
        db_dir = os.path.join("chroma_dbs", safe_name)

        if st.button("Process & Index Book", type="primary") or (
            os.path.exists(db_dir) and st.session_state.get("current_book") != safe_name
        ):
            with st.spinner("Reading and indexing book into ChromaDB..."):
                if os.path.exists(db_dir):
                    vectorstore = Chroma(
                        persist_directory=db_dir,
                        embedding_function=embedding_model
                    )
                else:
                    vectorstore = process_pdf(uploaded_file, db_dir)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["current_book"] = safe_name
                st.session_state["messages"] = []  # Reset chat for new book
            st.success(f"Ready to chat with **{uploaded_file.name}**!")

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document."
"""),
    ("human", """Context:
{context}

Question: {question}""")
])

# Chat Input
if query := st.chat_input("Ask a question about your uploaded book..."):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload a PDF book in the sidebar and click **Process & Index Book** first.")
    else:
        # 1. Show user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Retrieve context using MMR (same settings as your main.py)
        retriever = st.session_state["vectorstore"].as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching book and generating answer..."):
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)
                answer = response.text

                st.markdown(answer)

                # Optional: Show retrieved page sources in an expandable box
                with st.expander("View Retrieved Book Passages"):
                    for i, doc in enumerate(docs, 1):
                        page_num = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i} (Page {page_num}):** {doc.page_content}")

        st.session_state["messages"].append({"role": "assistant", "content": answer})