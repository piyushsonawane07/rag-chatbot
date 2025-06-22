import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

def get_separated_documents_by_type(folder_path):
    doc_types = {
        'pdf': [],
        'csv': [],
        'docx': [],
        'xlsx': []
    }

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            full_path = os.path.join(root, file)

            if ext == '.pdf':
                doc_types['pdf'].append(full_path)
            elif ext == '.csv':
                doc_types['csv'].append(full_path)
            elif ext == '.docx':
                doc_types['docx'].append(full_path)
            elif ext == '.doc':
                doc_types['doc'].append(full_path)
            elif ext == '.xml':
                doc_types['xml'].append(full_path)
            elif ext == '.xlsx':
                doc_types['xlsx'].append(full_path)

    return doc_types

def pdf_loader(pdf_files):
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        all_docs.extend(loader.load())
    return all_docs

def docx_loader(docx_files):
    all_docs = []
    for docx_file in docx_files:
        loader = Docx2txtLoader(docx_file)
        all_docs.extend(loader.load())
    return all_docs

def documents_loader():
    print("Loading documents...")
    folder_path = r"Insurance PDFs"
    doc_files = get_separated_documents_by_type(folder_path)
    pdf_data = pdf_loader(doc_files['pdf'])
    docx_data = docx_loader(doc_files['docx'])
    return [pdf_data, docx_data]



       
