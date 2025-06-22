# TalktonicGroq-with-FileAttachment
TalkTonic is a Streamlit-based AI chatbot that supports file uploads (PDF, image, and text), OCR-based text extraction, and interaction with local LLaMA models using Ollama.

## Features
- 🔍 Extracts text from PDF (with OCR fallback), JPG/PNG, and TXT
- 🧠 Analyzes extracted or edited text using Groq API Key (Lllama 3)
- 💬 Clean chat interface with themes and downloadable chat history

## Tech Stack
- Python
- Streamlit
- Tesseract OCR
- Groq API Key (Lllama 3)
- PDFPlumber

## Run Locally

```bash
streamlit run your_script.py
