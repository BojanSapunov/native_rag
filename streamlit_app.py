import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# ---------- CONFIG ----------
MODELS = ["llama3", "gemma"]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "faiss_index"

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 RAG Settings")
model_choice = st.sidebar.selectbox("Choose Ollama Model:", options=MODELS, index=0)
history_limit = st.sidebar.slider("Number of past replies to show:", min_value=1, max_value=10, value=5)
top_k = st.sidebar.slider("Top-k context chunks to retrieve:", min_value=1, max_value=20, value=5)
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- LOAD COMPONENTS ----------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def get_llm(model_name):
    return Ollama(model=model_name)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
llm = get_llm(model_choice)
qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

# ---------- MAIN APP ----------
st.title("📚 Connect.it Local RAG with Ollama + FAISS")
query = st.text_input("Ask a question based on your PDFs:")

if query:
    chat_context = st.session_state.chat_history[-history_limit:]
    chat_summary = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in chat_context])

    prompt = f"""
    [INST]
    You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question, use the chat history
    between <chat_history> and </chat_history> to provide a coherent and helpful answer.

    If the question can't be answered with the available information, respond with:
    "Sorry, I don't know the answer to that question."

    Avoid phrases like "according to the provided context".

    <chat_history>
    {chat_summary}
    </chat_history>
    <question>
    {query}
    </question>
    [/INST]
    Answer:
    """

    with st.spinner("Retrieving context and generating answer..."):
        docs = retriever.get_relevant_documents(query)
        result = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        answer = result["output_text"]

    # Save interaction
    st.session_state.chat_history.append({"question": query, "answer": answer})

    # Display answer
    st.markdown("### 🧠 Answer")
    st.write(answer)

    # Display sources
    st.markdown("### 📄 Retrieved Chunks")
    for doc in docs:
        meta = doc.metadata
        st.markdown(f"**📘 File:** {meta.get('source', 'Unknown')} — **Page:** {meta.get('page', '?')}")
        with st.expander("View Chunk Text"):
            st.code(doc.page_content)
