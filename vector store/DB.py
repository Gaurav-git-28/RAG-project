from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs=[
    Document(page_content="Python is widely used in AI",metadata={"source":"AI book"}),
    Document(page_content="Pandas is used for data analysis in python",metadata={"source":"DS book"}),
    Document(page_content="Neural network are used in deep learning",metadata={"source":"DL book"}),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    client_args={"http2": False}
)
vectorstore= Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="Chroma-db"
)

result=vectorstore.similarity_search("what is used for data analysis",k=2)

for r in result:
    print(r.page_content)

retriver=vectorstore.as_retriever()

docs=retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
    
