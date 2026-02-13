import os
import re
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -------------------------
# Gemini Setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment.")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# Embedding Model
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# In-memory storage
# -------------------------
documents = []
doc_chunks = []
doc_names = []
index = None


# -------------------------
# Smarter Chunking
# -------------------------
def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------
# Upload Document
# -------------------------
def add_document(file):
    global index, documents, doc_chunks, doc_names

    if file is None:
        return "Please upload a .txt file."

    # Reset workspace for clean demo behavior
    index = None
    documents = []
    doc_chunks = []
    doc_names = []

    try:
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    chunks = chunk_text(content)

    if len(chunks) == 0:
        return "Uploaded file is empty."

    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    clean_name = os.path.basename(file.name)

    for chunk in chunks:
        doc_chunks.append(chunk)
        doc_names.append(clean_name)

    documents.append(clean_name)

    return f"Workspace reset. Uploaded: {clean_name}"


# -------------------------
# List Documents
# -------------------------
def list_documents():
    if not documents:
        return "No documents uploaded."
    return "\n".join(documents)


# -------------------------
# Ask Question
# -------------------------
def ask_question(question):
    global index

    if index is None:
        return "Please upload a document first.", "", ""

    if question is None or question.strip() == "":
        return "Please enter a valid question.", "", ""

    query_embedding = embedder.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=5)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in I[0]:
        if idx < len(doc_chunks):
            retrieved_chunks.append(doc_chunks[idx])
            retrieved_sources.append(doc_names[idx])

    if not retrieved_chunks:
        return "No relevant content found.", "", ""

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict document-based question answering system.

Use ONLY the provided context.
Do NOT use outside knowledge.
If the answer is not clearly present in the context, say exactly:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    try:
        response = llm.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}", "", ""

    source_display = "\n".join(set(retrieved_sources))
    chunk_display = "\n\n---\n\n".join(retrieved_chunks)

    return answer, source_display, chunk_display


# -------------------------
# System Status
# -------------------------
def system_status():
    backend_status = "OK"
    db_status = "OK" if index is not None else "No documents loaded"

    try:
        llm.generate_content("Say OK")
        llm_status = "OK"
    except Exception:
        llm_status = "LLM connection failed"

    return f"Backend: {backend_status}\nVector DB: {db_status}\nLLM: {llm_status}"


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("""
# Mini Private Knowledge Q&A Workspace

Upload a text document, ask questions, and see exactly which document and text snippet the answer comes from.
""")

    with gr.Tab("Upload"):
        file_input = gr.File(file_types=[".txt"])
        upload_btn = gr.Button("Upload Document")
        upload_output = gr.Textbox()
        upload_btn.click(add_document, file_input, upload_output)

    with gr.Tab("Documents"):
        list_btn = gr.Button("Show Uploaded Documents")
        list_output = gr.Textbox()
        list_btn.click(list_documents, outputs=list_output)

    with gr.Tab("Ask"):
        question_input = gr.Textbox(label="Your Question")
        ask_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer")
        source_output = gr.Textbox(label="Source Documents")
        chunk_output = gr.Textbox(label="Relevant Chunks")
        ask_btn.click(
            ask_question,
            question_input,
            [answer_output, source_output, chunk_output]
        )

    with gr.Tab("System Status"):
        status_btn = gr.Button("Check Status")
        status_output = gr.Textbox()
        status_btn.click(system_status, outputs=status_output)

demo.launch()