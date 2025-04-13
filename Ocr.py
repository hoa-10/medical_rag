import easyocr
import cv2
import os
import numpy as np
from pdf2image import convert_from_path



def extract_text_from_pdf(pdf_path, poppler_path=None):
    # Create temp directory for images
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    reader = easyocr.Reader(['vi']) 
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)  
        all_text = ""
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            img = cv2.imread(image_path)
            results = reader.readtext(img)
            page_text = ""
            for (bbox, text, prob) in results:
                page_text += text + " "
            
            all_text += f"--- Page {i+1} ---\n{page_text}\n\n"
            os.remove(image_path)
        return all_text
    
    finally:
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass
#HEllo
#pdf_file = "11. HDSD.pdf"
poppler_path = r"C:\Users\Admin\Desktop\medical_rag\poppler-24.08.0\Library\bin" 
#extracted_text = extract_text_from_pdf(pdf_file, poppler_path)
#with open("extracted_text.txt", "w", encoding="utf-8") as f:
#    f.write(extracted_text)
#
#print("Text extraction completed!")