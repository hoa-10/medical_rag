from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import cv2
import glob
from Ocr import extract_text_from_pdf, poppler_path
from langchain.schema import Document

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs = {"device": "cpu"}
)

def process_all_pdf(pdf_dir, persist_dir='pdf_database'):
    """
    Process all PDFs in a directory and create/load a vector database for RAG.
    Handles both text-based PDFs and scanned document PDFs.
    """
    if os.path.exists(persist_dir):
        print("Loading existing vector database...")
        try:
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding,
                collection_name='pdf-rag-chroma'
            )
            print("Successfully loaded existing vector database")
            return vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Will create new vector database...")
    
    print("Creating new vector database from PDF files...")
    documents = load_documents_with_smart_extraction(pdf_dir)
    
    if not documents:
        print("No documents were processed successfully.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=25
    )
    doc_split = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(
        documents=doc_split,
        collection_name='pdf-rag-chroma',
        embedding=embedding,
        persist_directory=persist_dir
    )
    
    vector_store.persist()
    print("Successfully created and persisted new vector database")
    
    return vector_store.as_retriever(search_kwargs={'k': 5})

def load_documents_with_smart_extraction(pdf_dir):
    """
    Load all PDFs from directory using smart extraction that first attempts
    direct text extraction and falls back to OCR if needed.
    """
    documents = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        try:
            docs = extract_text_directly(pdf_path)
            
            if docs and len(docs) > 0:
                has_content = False
                for doc in docs:
                    if len(doc.page_content.strip()) > 1:
                        has_content = True
                        break
                
                if has_content:
                    documents.extend(docs)
                    print(f"✓ Used direct extraction for {os.path.basename(pdf_path)}")
                    continue  
            print(f"! Direct extraction insufficient, using OCR for {os.path.basename(pdf_path)}")
            ocr_text = extract_text_from_pdf(pdf_path, poppler_path)
            print(ocr_text)
            if ocr_text.strip(): 
                documents.append(Document(
                    page_content=ocr_text,
                    metadata={"source": pdf_path, "extraction_method": "ocr"}
                ))
                
        except Exception as e:
            print(f"! Error processing {pdf_path}: {e}")
            print("  Falling back to OCR-based extraction")
            try:
                ocr_text = extract_text_from_pdf(pdf_path, poppler_path)
                print(ocr_text)
                if ocr_text.strip(): 
                    documents.append(Document(
                        page_content=ocr_text,
                        metadata={"source": pdf_path, "extraction_method": "ocr_fallback"}
                    ))
            except Exception as e2:
                print(f"!! Failed to extract text from {pdf_path}: {e2}")
    
    return documents

def extract_text_directly(pdf_path):
    """Extract text directly from PDF using PyPDFLoader"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["extraction_method"] = "direct"
        
        return documents
    except Exception as e:
        print(f"Direct extraction error: {e}")
        return []



