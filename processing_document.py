import pdfplumber
import fitz  # PyMuPDF
import os
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from classification_pageType import is_text, is_table_of_contents, is_image
from llm_reasoning import analyze_image
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from translation_query_language import translate_vi2en
#model embedding 
text_embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    cache_folder=r"C:\Users\user\model_embedding"
)
COLLECTION_NAME = "pdf-documents"
def extract_text_from_pdf_page(pdf_path, page_number):
    """Trích xuất văn bản từ trang PDF"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 0 or page_number >= len(pdf.pages):
                return ""
            page = pdf.pages[page_number]
            text = page.extract_text() or ""
            return text.strip()
    except Exception as e:
        print(f"Lỗi khi trích xuất text: {e}")
        return ""

def convert_pdf_page_to_image(pdf_path, page_number):
    """Chuyển đổi trang PDF thành hình ảnh"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Lỗi khi chuyển PDF sang image: {e}")
        return None

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

def process_page(pdf_path, page_number, pdf_filename):
    """Xử lý một trang PDF và trả về dữ liệu cho vector database"""
    page_type = check_page_content(pdf_path, page_number)
    page_id = f"{pdf_filename}_page_{page_number}"
    
    print(f"Đang xử lý {pdf_filename} trang {page_number} - Loại: {page_type}")
    
    # Bỏ qua các trang lỗi
    if page_type == "Error" or page_type == "Unknown":
        return None
    
    # Xử lý các loại trang khác nhau
    content = ""
    if "Images" in page_type or "TOC + text" in page_type + "TOC + image + text" in page_type or "Text with image" in page_type:
        image = convert_pdf_page_to_image(pdf_path, page_number)
        if image:
            image_analysis = analyze_image(image)
            content += f"IMAGE ANALYSIS:\n{image_analysis}\n\n"
        if "Text" in page_type:
            text = extract_text_from_pdf_page(pdf_path, page_number)
            content += f"TEXT CONTENT:\n{text}"
    else:
        content = extract_text_from_pdf_page(pdf_path, page_number)
    

    if not content:
        return None
    
    # Trả về document để xử lý sau
    return Document(
        page_content=content,
        metadata={
            "id": page_id,
            "source": pdf_filename,
            "page": page_number,
            "type": page_type
        }
    )

def process_pdf_documents(pdf_dir, persist_dir="./chroma_db"):
    """Xử lý toàn bộ tài liệu PDF và tạo vector database"""
    try:
        if os.path.exists(persist_dir):
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=text_embedding_model,
                persist_directory=persist_dir
            )
            
            # Kiểm tra số lượng documents
            collection_size = len(vector_store.get())
            if collection_size > 0:
                print(f"Đã tìm thấy vector database với {collection_size} mục, đang tải...")
                return vector_store.as_retriever(search_kwargs={'k': 5})
            else:
                print("Vector database tồn tại nhưng trống, tiến hành embedding mới...")
    except Exception as e:
        print(f"Không thể tải vector database hiện có: {e}. Sẽ tạo mới.")
    
    # Xử lý mới nếu chưa có database hoặc database rỗng
    processed_docs = []
    
    # Xử lý từng file PDF
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    print(f"Đang xử lý {filename} với {num_pages} trang")
                    
                    for page_num in range(num_pages):
                        doc = process_page(pdf_path, page_num, filename)
                        if doc:
                            processed_docs.append(doc)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")
    
    print(f"Đã xử lý thành công {len(processed_docs)} trang")
    
    # Nếu không có tài liệu nào được xử lý
    if not processed_docs:
        print("Cảnh báo: Không có tài liệu nào được xử lý thành công")
        return None
    
    # Chia nhỏ các document
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=25
    )
    
    doc_chunks = text_splitter.split_documents(processed_docs)
    
    # Xóa collection cũ nếu tồn tại
    try:
        # Xóa cả collection trực tiếp từ ChromaDB
        client = chromadb.PersistentClient(persist_dir)
        collections = [col.name for col in client.list_collections()]
        if COLLECTION_NAME in collections:
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Lỗi khi xóa collection cũ: {e}")
    
    # Tạo vector store mới với collection thống nhất
    vector_store = Chroma.from_documents(
        documents=doc_chunks,
        embedding=text_embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir
    )
    vector_store.persist()
    print(f"Đã tạo và lưu vector database thành công với {len(doc_chunks)} chunks")
    
    return vector_store.as_retriever(search_kwargs={'k': 5})

def query_document(query_text, retriever=None, persist_dir="./chroma_database"):
    """Truy vấn vector database dựa trên câu hỏi"""
    query_user = translate_vi2en(query_text)
    print(query_user)
    if retriever is None:
        # Tải lại retriever nếu chưa được cung cấp
        try:
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=text_embedding_model,
                persist_directory=persist_dir
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        except Exception as e:
            print(f"Lỗi khi tải vector store: {e}")
            return []
    
    try:
        # Truy vấn các tài liệu liên quan
        docs = retriever.get_relevant_documents(query_user)
        return docs
    except Exception as e:
        print(f"Lỗi khi truy vấn vector store: {e}")
        return []

