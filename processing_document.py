import pdfplumber
import torch
import re
import os
from transformers import AutoProcessor
import numpy as np
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Tuple
import heapq
from colpali_engine.models import ColPali, ColPaliProcessor
import chromadb

# Initialize embedding model
text_embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    cache_folder="D:\\model_embedding"
)

# Initialize ColPali model for image processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vidore/colpali-v1.2"
colpali_model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.float32,

).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Initialize ChromaDB for vector storage
client = chromadb.PersistentClient("./chroma_db")
colpali_collection = client.get_or_create_collection("colpali_embeddings")
text_collection = client.get_or_create_collection("text_embeddings")

def is_text(pdf_path, page_number, min_text_length=10):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 0 or page_number >= len(pdf.pages):
                raise ValueError(f"Page {page_number} out of range. PDF has {len(pdf.pages)} pages.")
            page = pdf.pages[page_number]
            text = page.extract_text()
            return text and len(text.strip()) >= min_text_length
    except Exception as e:
        print(f"Error checking text: {e}")
        return False

def is_table_of_contents(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            tables = page.extract_tables()
            
            # If there are tables, return True
            if len(tables) > 0:
                return True
                
            # Check text structure that might be a table of contents
            text = page.extract_text() or ""
            if text:
                # Regex pattern to check for lines with "[text]...[number]" or "[number].[number]"
                # commonly found in table of contents
                toc_pattern = re.compile(r'(^|\n)[\w\s]+\.{2,}\s*\d+|^\d+\.\d+\s+[\w\s]+', re.MULTILINE)
                if len(re.findall(toc_pattern, text)) > 3:  # Minimum 3 items to be considered TOC
                    return True
            
            return False
    except Exception as e:
        print(f"Error checking table of contents: {e}")
        return False

def is_image(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 0 or page_number >= len(pdf.pages):
                raise ValueError(f"Page {page_number} out of range. PDF has {len(pdf.pages)} pages.")
            page = pdf.pages[page_number]
            images = page.images
            return len(images) > 0
    except Exception as e:
        print(f"Error checking images: {e}")
        return False

def check_page_content(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 0 or page_number >= len(pdf.pages):
                raise ValueError(f"Page {page_number} out of range. PDF has {len(pdf.pages)} pages.")

        has_text_content = is_text(pdf_path, page_number)
        is_toc = is_table_of_contents(pdf_path, page_number)
        has_image_content = is_image(pdf_path, page_number)

        if has_image_content and not has_text_content and not is_toc:
            return "Images"
        if is_toc and has_image_content and has_text_content:
            return "TOC + image + text"  
        if has_text_content and has_image_content and not is_toc:
            return "Text with image"  
        if is_toc and not has_image_content and has_text_content:
            return "TOC + text"  
        if has_text_content:
            return "Text"  

        return "Unknown"
    except Exception as e:
        print(f"Error classifying page: {e}")
        return "Error"

def convert_pdf_page_to_image(pdf_path, page_number):
    """Convert PDF page to image"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None

def extract_text_from_pdf_page(pdf_path, page_number):
    """Extract text from PDF page"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            return page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_with_colpali(image):
    """Process image with ColPali model and return embedding"""
    if image is None:
        return np.zeros(768)
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = colpali_model(**inputs)
    
    # Get embedding from model
    image_embeddings = outputs.image_embeds.cpu().numpy()
    return image_embeddings[0]

def process_query_with_colpali(query):
    """Process query with ColPali model and return embedding"""
    try:
        # Use previously initialized ColPaliProcessor
        batch_queries = processor.process_queries([query]).to(device)
        
        with torch.no_grad():
            # Forward pass for query embedding
            query_outputs = colpali_model.forward(**batch_queries)
            
            # Get embedding from model
            query_embeddings = query_outputs.query_embeds.cpu().numpy()
        
        return query_embeddings[0]
    except Exception as e:
        print(f"Error processing query with ColPali: {e}")
        # Return empty vector if error
        return np.zeros(768)
def process_and_store_pdf_page(pdf_path, page_number):
    """Classify and process PDF page for each case"""
    try:
        content_type = check_page_content(pdf_path, page_number)
        page_id = f"{os.path.basename(pdf_path)}_page_{page_number}"
        extracted_text = extract_text_from_pdf_page(pdf_path, page_number)
        
        print(f"Processing page {page_number}: {content_type}")
        
        # Create common metadata
        metadata = {
            "content_type": content_type,
            "page": page_number,
            "pdf": pdf_path,
            "text": extracted_text[:1000]  # Store start of text for display
        }
        
        # Only process with ColPali for cases with images or complex requirements
        if "Images" in content_type or "TOC + image + text" in content_type or "Text with image" in content_type or "TOC + text" in content_type:
            img = convert_pdf_page_to_image(pdf_path, page_number)
            if img:
                colpali_embedding = process_with_colpali(img)
                
                try:
                    colpali_collection.add(
                        embeddings=[colpali_embedding.tolist()],
                        metadatas=[metadata],
                        ids=[f"colpali_{page_id}"]
                    )
                except Exception as e:
                    print(f"Error saving colpali embedding: {e}")
        
        # If text or has text, save text embedding
        if "Text" in content_type:
            text_embedding = text_embedding_model.embed_query(extracted_text)
            
            try:
                text_collection.add(
                    embeddings=[text_embedding.tolist()],
                    metadatas=[metadata],
                    ids=[f"text_{page_id}"]
                )
            except Exception as e:
                print(f"Error saving text embedding: {e}")
        
        return True
    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        return False

def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:

    try:
        # Create embedding for query
        text_query_embedding = text_embedding_model.embed_query(query)
        colpali_query_embedding = process_query_with_colpali(query)
        
        # Search with text embedding
        text_results = text_collection.query(
            query_embeddings=[text_query_embedding.tolist()],
            n_results=top_k
        )
        
        # Search with ColPali embedding
        colpali_results = colpali_collection.query(
            query_embeddings=[colpali_query_embedding.tolist()],
            n_results=top_k
        )
        
        # Combine results
        combined_results = []
        
        # Add results from text search
        if text_results and 'metadatas' in text_results and text_results['metadatas']:
            for i, (metadata, distance) in enumerate(zip(text_results['metadatas'][0], text_results['distances'][0])):
                combined_results.append({
                    'metadata': metadata,
                    'similarity': float(distance),
                    'source': 'text'
                })
        
        # Add results from image search
        if colpali_results and 'metadatas' in colpali_results and colpali_results['metadatas']:
            for i, (metadata, distance) in enumerate(zip(colpali_results['metadatas'][0], colpali_results['distances'][0])):
                combined_results.append({
                    'metadata': metadata,
                    'similarity': float(distance),
                    'source': 'colpali'
                })
        
        # Sort results by similarity (descending)
        sorted_results = sorted(combined_results, key=lambda x: x['similarity'], reverse=True)
        
        # Remove duplicates (based on pdf and page)
        unique_results = []
        seen_pages = set()
        
        for result in sorted_results:
            key = (result['metadata']['pdf'], result['metadata']['page'])
            if key not in seen_pages:
                seen_pages.add(key)
                unique_results.append(result)
                
                # Stop when we have top_k results
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    except Exception as e:
        print(f"Error performing hybrid search: {e}")
        return []


def process_pdf(pdf_path, start_page=0, end_page=None):
    """Process multiple pages in a PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if end_page is None:
                end_page = total_pages
            
            for page_num in range(start_page, min(end_page, total_pages)):
                success = process_and_store_pdf_page(pdf_path, page_num)
                if success:
                    print(f"Processed page {page_num}/{total_pages}")
                else:
                    print(f"Could not process page {page_num}/{total_pages}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

def load_documents_with_smart_extraction(pdf_dir):
    """Load documents from PDFs with smart extraction based on page content"""
    all_documents = []
    
    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing PDF: {pdf_file}")
        
        # Process all pages in the PDF and store in vector databases
        process_pdf(pdf_path)
        
        # Now create Document objects for LangChain from the processed pages
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num in range(total_pages):
                    content_type = check_page_content(pdf_path, page_num)
                    extracted_text = extract_text_from_pdf_page(pdf_path, page_num)
                    
                    # Create metadata
                    metadata = {
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": content_type,
                    }
                    
                    # Create Document object based on content type
                    if extracted_text:
                        all_documents.append(Document(  
                            page_content=extracted_text,
                            metadata=metadata
                        ))
                    elif content_type == "Images" or content_type == "TOC + image + text" or content_type == "Text with image" or content_type == "TOC + text":
                        # For image-only pages with no text
                        metadata["content_type"] = "Image page"
                        all_documents.append(Document(
                            page_content=f"[Image on page {page_num+1}]",
                            metadata=metadata
                        ))
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    print(f"Loaded {len(all_documents)} document chunks from {len(pdf_files)} PDF files")
    return all_documents

class HybridRetriever:
    """Custom retriever that uses both text and image embeddings"""
    
    def __init__(self, k=5):
        self.k = k
    
    def get_relevant_documents(self, query_str):
        """Find relevant documents using hybrid search"""
        results = hybrid_search(query_str, self.k)
        
        # Convert to Document objects for compatibility with LangChain
        documents = []
        for result in results:
            # Get the PDF path and page number
            pdf_path = result['metadata']['pdf']
            page_num = result['metadata']['page']
            
            # Extract text for this page
            page_text = extract_text_from_pdf_page(pdf_path, page_num)
            
            # Create a Document object
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "content_type": result['metadata']['content_type'],
                    "similarity": result['similarity'],
                    "search_type": result['source']
                }
            )
            documents.append(doc)
        
        return documents

def process_all_filepdf(pdf_dir, persist_dir='pdf_database', use_hybrid=True):

    # If hybrid search is requested, use our custom retriever
    if use_hybrid:
        print("Initializing hybrid search retriever...")
        # Process all PDFs to ensure they're in our vector stores
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing PDF for hybrid search: {pdf_file}")
            process_pdf(pdf_path)
            
        # Return our hybrid retriever
        return HybridRetriever(k=5)
    
    # Otherwise, use the standard LangChain Chroma retriever
    if os.path.exists(persist_dir):
        print("Loading existing vector database...")
        try:
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=text_embedding_model,
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
        embedding=text_embedding_model,
        persist_directory=persist_dir
    )
    
    vector_store.persist()
    print("Successfully created and persisted new vector database")
    
    return vector_store.as_retriever(search_kwargs={'k': 5})




# Helper function to check database status
def get_database_stats():
    """Get information about the number of vectors stored in the database"""
    try:
        text_count = text_collection.count()
        colpali_count = colpali_collection.count()
        
        return {
            "text_vectors": text_count,
            "colpali_vectors": colpali_count,
            "total": text_count + colpali_count
        }
    except Exception as e:
        print(f"Error getting database information: {e}")
        return {"error": str(e)}