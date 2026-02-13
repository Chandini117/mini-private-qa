---
title: Mini Private Qa
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# Mini Private Knowledge Q&A Workspace

This is a lightweight document-based question answering web application.

## What It Does

- Upload a `.txt` document
- Ask natural language questions
- Retrieve answers grounded only in the uploaded document
- Display the exact source document and relevant text snippets
- Show system health status (Backend, Vector DB, LLM)

## How It Works

1. The uploaded document is split into sentence-based chunks.
2. Each chunk is converted into embeddings using `all-MiniLM-L6-v2`.
3. Embeddings are stored in a FAISS index.
4. When a question is asked:
   - Top relevant chunks are retrieved.
   - Retrieved context is sent to Gemini.
   - The model answers strictly using that context.

## Tech Stack

- Gradio (UI)
- FAISS (Vector Search)
- Sentence Transformers (Embeddings)
- Gemini 2.5 Flash (LLM)

## Environment Variable Required

GEMINI_API_KEY

## Limitations

- Only `.txt` files supported
- No persistent storage (workspace resets on new upload)
- Not production-scaled (demo purpose)

This project focuses on core Retrieval-Augmented Generation (RAG) behavior.