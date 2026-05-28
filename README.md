# Flight Manual RAG Chatbot

An AI-powered Q&A chatbot for aviation documents using a real **Retrieval-Augmented Generation (RAG)** pipeline — built with Python, FastAPI, ChromaDB, and Groq (free!).

Upload any flight manual PDF (POH, AFM, maintenance manual) and ask questions — the system retrieves only the most relevant sections before answering, grounded in your actual document.

<img width="1600" height="721" alt="demo2" src="https://github.com/user-attachments/assets/78eb0e9a-285e-4a24-a354-b046a903146e" />
<img width="1600" height="727" alt="demo1" src="https://github.com/user-attachments/assets/b1470927-d216-4ab8-86d7-c997b2fdf073" />
---

## How the RAG Pipeline Works
PDF Upload
↓
pypdf extracts all text from every page
↓
Text is split into ~500-word chunks (with 50-word overlap)
↓
ChromaDB embeds & indexes all chunks (vector database)
↓
User asks a question
↓
ChromaDB finds the top 5 most semantically similar chunks
↓
Groq LLaMA 3.3 70B answers using ONLY those chunks

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+ + FastAPI |
| PDF Parsing | pypdf |
| Vector Database | ChromaDB |
| LLM | LLaMA 3.3 70B via Groq API (free) |
| Frontend | Vanilla HTML + CSS + JS |

---

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/AryaTawade/flight-manual-rag.git
cd flight-manual-rag
```

### 2. Set up Python environment
```bash
cd backend
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# If you get a numpy/chromadb conflict:
pip install numpy==1.26.4 chromadb==0.6.3

# If you get a groq proxies error:
pip install groq --upgrade httpx --upgrade
```

### 4. Get your FREE Groq API key
1. Go to https://console.groq.com
2. Sign up (free, no credit card needed)
3. Create an API key

```bash
# Windows:
echo GROQ_API_KEY=your_key_here > .env

# Mac/Linux:
echo "GROQ_API_KEY=your_key_here" > .env
```

### 5. Start the backend
```bash
uvicorn main:app --reload
```
Server runs at: http://127.0.0.1:8000

### 6. Open the frontend
Open `frontend/index.html` directly in your browser or use the VS Code Live Server extension.

---

## Project Structure
flight-manual-rag/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── .env
├── .gitignore
└── README.md


---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /upload | Upload PDF → chunk → embed into ChromaDB |
| POST | /chat | RAG query → retrieve chunks → LLaMA answer |
| GET | /sessions/{id} | Get session info |
| DELETE | /sessions/{id} | Clear session |

---








