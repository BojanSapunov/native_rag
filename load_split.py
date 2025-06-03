from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os

# Load all PDF files in a folder
pdf_folder = "data/pdfs"
all_docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Loading {pdf_path}...")
        # Load the PDF file
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_docs.extend(documents)

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(all_docs)


# Generate embeddings for the text chunks
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [embed_model.encode(doc.page_content) for doc in docs]

# Create a FAISS vector store from the embeddings and store the document chunks
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_function)

# Initialize the LLM and create a RetrievalQA chain
llm = Ollama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

query = "What is the key question regarding deployment for enterprises with data maturity level four?"
print(qa.run(query))
