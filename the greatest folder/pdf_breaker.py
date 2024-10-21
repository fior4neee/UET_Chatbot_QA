import os
import pandas as pd
import requests
import pytesseract
from pdf2image import convert_from_path
from tqdm.auto import tqdm
files_to_break = ["QA/27-7-2022-Du thao Quy che TS.doc",
                  "QA/1533.VNU_Kế hoạch triển khai công tác tuyển sinh ĐHCQ năm 2024 của ĐHQGHN.pdf",
                  "QA/3626_21.10.2022. Quy chế đào tạo đại học tại ĐHQGHN (áp dụng từ khóa QH2022).docx",
                  "QA/4555 Quy dinh mo nganh va dieu chinh CTĐT tại ĐHQGHN.doc",
                  "QA/CV số 1957 Hướng dẫn TS năm 2024.pdf",
                  "QA/Du thao Quy che dao tao ThS 2022 V1.docx",
                  "QA/QHI_Đề án TS ĐHCQ năm 2024 (Điều chỉnh).doc",
                  "QA/Signed.Signed.Signed.Signed.Signed.Quyết định ban hành Quy chế tuyển sinh đại học chính quy tại ĐHQGHN (trình ký) (1).pdf",
                  "QA/Signed.Signed.Signed.Signed.Signed.Quyết định ban hành Quy chế tuyển sinh đại học chính quy tại ĐHQGHN (trình ký).docx",
                  "QA/V10_Dự thảo quy định VB CC CN -trình ký 15.4.2023.docx"]
def text_formatter(text:str) -> str:
  cleaned_text = text.replace("-\n"," ").strip()
  # More text formatting can go here
  return cleaned_text

import pytesseract
from pdf2image import convert_from_path
from tqdm.auto import tqdm

# 8 - 10 it/s
# 8 - 9 mins on GPU

def text_formatter(text: str) -> str:
    """Format and clean extracted text."""
    return text.replace("-\n", " ").strip()

def open_and_read_image_pdf(pdf_path: str) -> list[dict]:
    """Convert PDF pages to images and extract text using OCR."""
    images = convert_from_path(pdf_path)
    page_from = pdf_path.split("/")[-1].split(".pdf")[0]
    pages_and_texts = []

    for i, image in enumerate(tqdm(images)):
        text = pytesseract.image_to_string(image)
        text = text_formatter(text=text)
        pages_and_texts.append({
            "page_number": i + 1,
            "page_from": page_from,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text) / 4,  # Assuming 1 token = ~4 characters
            "text": text
        })

    return pages_and_texts

def process_pdf_files(pdf_files: list[str]) -> list[dict]:
    """Process multiple PDF files."""
    all_pages_and_texts = []
    for pdf_path in pdf_files:
        if pdf_path.endswith(".pdf"):
            print(f"Processing PDF: {pdf_path}")
            pages_and_texts = open_and_read_image_pdf(pdf_path=pdf_path)
            all_pages_and_texts.extend(pages_and_texts)  # Combine all results
    return all_pages_and_texts

# Process all PDFs
all_pages_and_texts = process_pdf_files(files_to_break)

for page_data in all_pages_and_texts:
    if page_data['text'] is not str:
      print(page_data['text'])

all_pages_and_texts

def convert_floats_to_strings(data: dict) -> dict:
    """Converts all float values in the dictionary to strings."""
    for key, value in data.items():
        if isinstance(value, float):
            data[key] = str(value)
    return data
all_pages_and_texts = [
    {k: str(v) if isinstance(v, float) else v for k, v in d.items()}
    for d in all_pages_and_texts
]

all_pages_and_texts_df = pd.DataFrame(all_pages_and_texts)
all_pages_and_texts_df