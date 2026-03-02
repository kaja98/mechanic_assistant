import sys 
import os
from pathlib import Path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 
if ROOT not in sys.path: 
    sys.path.insert(0, ROOT)
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from src.main import run_pipeline
from src.utils import parse_source_info


BASE_DIR = Path(__file__).resolve().parent # src/studio 
PDF_DIR = BASE_DIR / "pdfs"

@st.cache_data
def list_pdf_files(directory: Path) ->str:
    """List all PDF files in the directory"""
    pdf_files = list(directory.glob("*.pdf"))
    return [file.name for file in pdf_files]

@st.cache_data
def load_pdf(file_path: Path) -> bytes:
    """Load and display PDF files."""
    if not file_path.exists():
        return None
    with file_path.open("rb") as file:
        pdf: bytes = file.read()
    return pdf
    
def main():
    st.set_page_config(page_title="Mechanic Assistant", page_icon="🔧", layout="wide")

    st.title("🔧 Mechanic Assistant Chatbot")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_binary" not in st.session_state:
        st.session_state.pdf_binary = None

    # Layout: Chat left, PDF right
    chat_col, pdf_col = st.columns([1, 1])

    with chat_col:

        # Chat input
        if prompt := st.chat_input("Ask a question about your machine..."):
            st.session_state.messages.insert(0, {"role": "user", "content": prompt})


            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = run_pipeline(prompt)

            st.session_state.messages.insert(0, {"role": "assistant", "content": answer})

            # Extract source info
            info = parse_source_info(answer)
            st.session_state.last_source_info = info
        
        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    # --- PDF VIEWER ---
    with pdf_col:
        st.subheader("📄 Source Document")

        info = st.session_state.get("last_source_info")

        if info:
            source = info["source"]
            page = info["page"]

            # Map source → PDF file path
            pdf_map = {
                "LAD-Front-Loading-Service-Manual-L11": Path(PDF_DIR / "LAD-Front-Loading-Service-Manual-L11.pdf"),
                "technical-manual-w11663204-revb":Path(PDF_DIR / "technical-manual-w11663204-revb.pdf")
            }

            pdf_path = pdf_map.get(source)

            if pdf_path:
                pdf_bytes = load_pdf(pdf_path)
                if pdf_bytes:
                    st.write(f"Showing: **{source}**, page **{page}**")
                    pdf_viewer(pdf_bytes, width=600, height=800, scroll_to_page=page)
                
            else:
                st.warning(f"PDF not found for source: {source}")
        else:
            st.info("Ask a question to load the relevant manual page.")

if __name__ == "__main__":
    main()
