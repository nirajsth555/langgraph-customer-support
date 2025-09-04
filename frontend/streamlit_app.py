import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Change if your FastAPI is hosted elsewhere

st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")

st.title("📄 AI-Powered PDF Question Answering")
st.markdown("Upload a PDF and ask questions about its content.")

# --- Upload PDF Section ---
st.header("1️⃣ Upload PDF")
with st.form("upload_form"):
    pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
    upload_btn = st.form_submit_button("Upload")

    if upload_btn:
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload_pdf/", files=files)
                res.raise_for_status()
                st.success(f"{pdf_file.name} uploaded successfully.")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

# --- Ask Question Section ---
st.header("2️⃣ Ask a Question")
with st.form("ask_form"):
    question = st.text_area("Your Question")
    ask_btn = st.form_submit_button("Ask")

    if ask_btn:
        if not question:
            st.warning("Please enter a question.")
        else:
            try:
                res = requests.post(
                    f"{BACKEND_URL}/ask/",
                    data={"question": question}
                )
                res.raise_for_status()
                answer = res.json()
                st.subheader("🧠 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error retrieving answer: {str(e)}")
