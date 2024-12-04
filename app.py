import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Chatbot with Groq",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_pdf(pdf_file):
    """Read and extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def build_qa_chain(text):
    """Build a QA chain using Groq and FAISS for conversational retrieval."""
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Local embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize Groq chat model
    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768"
    )

    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_chat,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return qa_chain

# Streamlit application
def main():
    st.title("📄 PDF Chatbot with Groq")
    st.write("Upload a PDF, and start asking questions about its content!")

    # Sidebar for user instructions
    with st.sidebar:
        st.header("Instructions")
        st.write(
            """
            - Upload a PDF document.
            - Enter your question in the input box.
            - Get answers specific to the content of the uploaded PDF.
            """
        )

    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    # Process the uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            pdf_text = load_pdf(uploaded_file)
            qa_chain = build_qa_chain(pdf_text)
            st.success("PDF successfully processed! Ask me anything about it.")

        # Chat interface
        st.markdown("### 💬 Ask Your Question")
        user_question = st.text_input("Type your question here", key="user_input")

        if user_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = qa_chain.run(user_question)
                    st.markdown(f"**🤖 Answer:** {response}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
