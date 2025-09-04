from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-ada-002")
