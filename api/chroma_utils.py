# chroma_utils.py

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader,
    JSONLoader, CSVLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
import os
import logging

logging.basicConfig(level=logging.INFO)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def load_and_split_document(file_path: str) -> List[Document]:
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    elif ext == 'html' or ext == 'htm':
        loader = UnstructuredHTMLLoader(file_path)
    elif ext == 'json':
        loader = JSONLoader(file_path)
    elif ext == 'csv':
        loader = CSVLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata['file_id'] = file_id
        
        vectorstore.add_documents(splits)
        vectorstore.persist()
        return True
    except Exception as e:
        logging.error(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        logging.info(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore.delete(where={"file_id": file_id})
        logging.info(f"Deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        logging.error(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
