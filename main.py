import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_DIR = "chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    client_args={"http2": False}
)

# 1. Create the database only if it doesn't already exist on disk
if not os.path.exists(DB_DIR):
    print("Loading and indexing PDF...")
    data = PyPDFLoader("document loaders/adobe.pdf")
    docs = data.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
else:
    print("Loading existing Chroma database...")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

# 2. Create the MMR retriever AFTER vectorstore is ready
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    client_args={"http2": False}
)

# 3. Fix ChatPromptTemplate.from_messages and closing quotes
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document."
"""),
    ("human", """Context:
{context}

Question: {question}""")
])

print("\nRAG System Ready! Press 0 to exit.")

while True:
    query = input("\nYou: ")
    if query.strip() == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
    print(f"\nAI: {response.text}")