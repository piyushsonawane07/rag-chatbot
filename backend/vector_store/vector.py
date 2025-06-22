from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders.docs.doc_loader import documents_loader
from loaders.web.web_loader import load_website
from uuid import uuid4

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Vector stores
# website_vs = Chroma(
#     collection_name="website_docs",
#     embedding_function=embeddings,
#     persist_directory="./chroma_website_db",
# )

# pdf_vs = Chroma(
#     collection_name="pdf_docs",
#     embedding_function=embeddings,
#     persist_directory="./chroma_pdf_db",
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

def store_documents():
    print("Storing documents...")
    documents = []
    pdf_documents, docx_documents = documents_loader()
    website_documents = load_website()
    documents.extend(pdf_documents)
    documents.extend(docx_documents)
    documents.extend(website_documents)

    split_docs = text_splitter.split_documents(documents)

    uuids_docs = [str(uuid4()) for _ in range(len(split_docs))]

    batch_size = 500
    for i in range(0, len(split_docs), batch_size):
        batch_docs = split_docs[i:i + batch_size]
        batch_uuids = uuids_docs[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_uuids)

    print("Documents stored successfully")
    print(f"{len(split_docs)} document chunks stored successfully.")

# def store_documents():
#     print("Storing documents...")
#     documents = []
#     pdf_documents, docx_documents = documents_loader()
#     documents.extend(pdf_documents)
#     documents.extend(docx_documents)
    
#     website_documents = load_website()

#     split_docs = text_splitter.split_documents(documents)
#     split_website_docs = text_splitter.split_documents(website_documents)

#     uuids_docs = [str(uuid4()) for _ in range(len(split_docs))]
#     uuids_website_docs = [str(uuid4()) for _ in range(len(split_website_docs))]

#     batch_size = 500
#     for i in range(0, len(split_docs), batch_size):
#         batch_docs = split_docs[i:i + batch_size]
#         batch_uuids = uuids_docs[i:i + batch_size]
#         pdf_vs.add_documents(documents=batch_docs, ids=batch_uuids)

#     for i in range(0, len(split_website_docs), batch_size):
#         batch_docs = split_website_docs[i:i + batch_size]
#         batch_uuids = uuids_website_docs[i:i + batch_size]
#         website_vs.add_documents(documents=batch_docs, ids=batch_uuids)

#     print("Documents stored successfully")
#     print(f"{len(split_docs)} document chunks stored successfully.")
#     print(f"{len(split_website_docs)} website document chunks stored successfully.")

# store_documents()
