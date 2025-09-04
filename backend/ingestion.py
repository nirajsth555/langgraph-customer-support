import os
import shutil
from fastapi import UploadFile
from utils import load_pdf_text, get_embedding_function
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

PERSIST_DIR = "vector_store"

# ----------------------------
# PDF Ingestion
# ----------------------------
async def ingest_pdf(file: UploadFile):
    os.makedirs("uploaded_files", exist_ok=True)
    file_path = f"uploaded_files/{file.filename}"

    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = load_pdf_text(file_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = splitter.split_text(text)

    # Embeddings + Chroma persistence
    embeddings = get_embedding_function()
    os.makedirs(PERSIST_DIR, exist_ok=True)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="default"
    )

    return {"status": "success", "message": f"{file.filename} ingested successfully."}

# ----------------------------
# Helpers for QA pipeline
# ----------------------------
def load_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="default"
    )

def _documents_from_chroma(vectorstore: Chroma):
    data = vectorstore.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data.get("metadatas", [{}] * len(docs))
    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(docs, metas)
    ]

# Cache hybrid retriever (to avoid rebuilding every request)
_HYBRID_RETRIEVER = None

def build_hybrid_retriever(vectorstore: Chroma, use_cache: bool = True):
    global _HYBRID_RETRIEVER
    if use_cache and _HYBRID_RETRIEVER is not None:
        return _HYBRID_RETRIEVER

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    documents = _documents_from_chroma(vectorstore)
    sparse_retriever = BM25Retriever.from_documents(documents)

    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )

    _HYBRID_RETRIEVER = hybrid
    return hybrid

def retrieve_context(vectorstore_or_hybrid, question: str, k: int = 3):
    if hasattr(vectorstore_or_hybrid, "get_relevant_documents"):
        hybrid = vectorstore_or_hybrid
    else:
        hybrid = build_hybrid_retriever(vectorstore_or_hybrid)
    docs = hybrid.invoke(question)
    return docs[:k]

def prompting(context, question: str) -> str:
    context_text = "\n".join([doc.page_content for doc in context])
    return f"""
You are an AI-powered customer support assistant. 
Your goal is to provide accurate, professional, and empathetic responses to customer queries 
using strictly the information provided in the context.

---

How to Think (Reasoning Process - Do NOT show this reasoning to the customer):
1. Understand the context: Carefully read the provided documents.
2. Locate relevance**: Identify which parts of the context are most relevant to the customer’s question.
3. Reason step-by-step**: Internally, think through how the relevant context answers the question. 
   - Break down complex context into simple explanations.
   - Compare multiple context points if needed.
4. Validate: Ensure your reasoning does not rely on outside or fabricated knowledge.
5. Formulate answer: Convert the reasoning into a clear, concise, and customer-friendly final response.

---

How to Respond (What the customer sees):
- Start with a polite and professional tone.
- Give a clear and concise answer** directly addressing the customer’s question.
- If necessary, provide a short explanation drawn from the context to make the answer more helpful.
- If the context does not contain the answer, politely say:
  *“I wasn’t able to find that information in the provided context.”*

---

IMPORTANT RULES:
1. You MUST only use the information present in the context.
2. You MUST NOT use any outside knowledge.
3. If the answer is not present in the context, respond exactly:
   "I’m sorry, the information is not available in the provided documents."
4. You MUST NOT fabricate, guess, or infer answers.

Context:
{context_text}

Customer Question:
{question}

---

Final Answer:
"""
