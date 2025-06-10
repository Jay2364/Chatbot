import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import shutil
import time

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# App layout
st.title("ELIZA")

# Upload section
st.sidebar.header("Upload and Embed Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize session states
if "embeddings" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "docs" not in st.session_state:
    st.session_state.docs = []

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to embed documents
def vector_embedding(uploaded_files):
    new_docs = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(file_path)
        new_docs.extend(loader.load())

    st.session_state.docs += new_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = text_splitter.split_documents(new_docs)

    if st.session_state.vectors is None:
        st.session_state.vectors = FAISS.from_documents(new_chunks, st.session_state.embeddings)
    else:
        new_vectors = FAISS.from_documents(new_chunks, st.session_state.embeddings)
        st.session_state.vectors.merge_from(new_vectors)

# Handle file upload
if uploaded_files:
    if not os.path.exists("temp_uploads"):
        os.makedirs("temp_uploads")

    if st.sidebar.button("Embed Documents"):
        vector_embedding(uploaded_files)
        st.sidebar.success("Documents have been embedded!")

# Chat interface
st.markdown("---")
st.subheader("Chat Interface")

# Initialize the ChatGroq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

# Input text for user messages
prompt1 = st.text_input("Type your message here...")

if prompt1:
    # Create document and retrieval chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever() if st.session_state.vectors else None
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1}) if retriever else {"answer": "No documents embedded yet."}
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # Store the conversation
    st.session_state.conversation.append({"user": prompt1, "bot": response['answer']})

    # Display conversation history in a styled format
    st.subheader("Previous Conversation")
    for chat in st.session_state.conversation:
        user_message = f'''
            <div style="
                text-align:left;
                padding:10px;
                background-color:rgb(5, 118, 54);
                border-radius:10px;
                max-width:70%;
                margin-bottom:10px;">
                👦🏻 {chat["user"]}
            </div>
        '''
        bot_message = f'''
            <div style="
                text-align:left;
                padding:10px;
                background-color: #333333;
                border-radius:10px;
                max-width:70%;
                margin-left:auto;
                margin-bottom:10px;">
                🤖 {chat["bot"]}
            </div>
        '''
        st.markdown(user_message, unsafe_allow_html=True)
        st.markdown(bot_message, unsafe_allow_html=True)

    with st.expander("Document Similarity Search"):
        if retriever:
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")
        else:
            st.write("No context to display.")

# Cleanup temporary files after use
if os.path.exists("temp_uploads"):
    shutil.rmtree("temp_uploads")
