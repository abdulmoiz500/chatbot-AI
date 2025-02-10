import streamlit as st
import time
from app import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

def upload_documents():
    st.title("Upload and Process PDFs")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    if st.button("Process"):
        if not pdf_docs:
            st.error("Please upload at least one PDF.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing..."):
            total_steps = 4  # We define 4 main steps
            
            # Step 1: Extract text from PDFs
            status_text.text("Extracting text from PDFs...")
            raw_text = get_pdf_text(pdf_docs)
            progress_bar.progress(25)  # 25% done
            time.sleep(1)  # Simulating processing delay

            # Step 2: Split text into chunks
            status_text.text("Splitting text into chunks...")
            text_chunks = get_text_chunks(raw_text)
            progress_bar.progress(50)  # 50% done
            time.sleep(1)  

            # Step 3: Generate vector store
            status_text.text("Creating vector store...")
            vectorstore = get_vectorstore(text_chunks)
            progress_bar.progress(75)  # 75% done
            time.sleep(1)  

            # Step 4: Initialize conversation chain
            status_text.text("Initializing conversation model...")
            st.session_state.conversation = get_conversation_chain(vectorstore)
            progress_bar.progress(100)  # 100% done

        st.success("Documents processed successfully! You can now start asking questions.")

if __name__ == '__main__':
    upload_documents()
