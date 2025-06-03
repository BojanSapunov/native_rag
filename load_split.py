from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Load all PDF files in a folder
pdf_folder = "data/pdfs"
all_docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        print(f"Loading {pdf_folder}/{filename}...")
        loader = UnstructuredPDFLoader(os.path.join(pdf_folder, filename))
        pages = loader.load()
       
        for i, page in enumerate(pages):
            page.metadata["source"] = filename
            page.metadata["page"] = i + 1
            all_docs.append(page)

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
docs = text_splitter.split_documents(all_docs)


# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and save FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
