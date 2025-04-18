import pdfplumber
import re
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
            if len(tables) > 0:
                return True
            text = page.extract_text() or ""
            if text:
                toc_pattern = re.compile(r'(^|\n)[\w\s]+\.{2,}\s*\d+|^\d+\.\d+\s+[\w\s]+', re.MULTILINE)
                if len(re.findall(toc_pattern, text)) > 3: 
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