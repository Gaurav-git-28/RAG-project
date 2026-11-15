from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document loaders/adobe.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks=splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    client_args={"http2": False}
)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)