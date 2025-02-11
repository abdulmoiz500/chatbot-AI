import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write(css, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        role = "You" if message["role"] == "user" else "Reply"
        st.markdown(f"**{role}**")
        st.markdown(f"{message['content']}")

    # User input
    user_question = st.text_input("Ask a question...", key="user_input")

    if user_question:
        # Store user input and clear the text input
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.user_input = ""  # Clear input field

        # Process and generate a response
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            bot_reply = response['chat_history'][-1].content  # Get the latest bot response
            st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

        # Reset the text input field to avoid infinite loop
        st.experimental_rerun()


if __name__ == '__main__':
    main()
