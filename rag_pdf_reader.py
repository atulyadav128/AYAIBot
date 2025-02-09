import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings


# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing! Add it to the .env file.")

# Load PDFs from data/ folder
def load_pdfs(folder_path="data/"):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {folder_path}")
    
    all_documents = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        all_documents.extend(loader.load())
    
    return all_documents

# Split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Create FAISS vector store
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index
    return vector_store

# Load FAISS and create retrieval chain
def create_qa_system():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

# Main function
def main():
    pdf_folder = "data/"
    
    if not os.path.exists("faiss_index"):
        print("Processing PDFs and creating vector database...")
        documents = load_pdfs(pdf_folder)
        chunks = split_text(documents)
        create_vector_store(chunks)
    else:
        print("Loading existing FAISS index...")

    qa_chain = create_qa_system()

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print("\nAnswer:", response)

if __name__ == "__main__":
    main()
