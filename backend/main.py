import os
import uuid
import io
import chromadb
from pypdf import PdfReader
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Flight Manual RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
chroma_client = chromadb.Client()

sessions: dict = {}


def extract_text_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    words = full_text.split()
    chunks, size, overlap = [], 500, 50
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


@app.post("/upload")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    chunks = extract_text_from_pdf(pdf_bytes)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    session_id = str(uuid.uuid4())
    collection = chroma_client.create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    sessions[session_id] = {
        "collection": collection,
        "filename": file.filename,
        "chunk_count": len(chunks),
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "message": f"Processed {len(chunks)} chunks from {file.filename}",
    }


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: List[dict] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a manual first.")

    collection = sessions[req.session_id]["collection"]

    results = collection.query(
        query_texts=[req.question],
        n_results=min(5, collection.count()),
    )
    relevant_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = f"""You are an expert aviation technical assistant for pilots and ground crew.

Answer ONLY using the flight manual excerpts below. Be precise and safety-focused.
If the answer is not in the excerpts, say so clearly — never guess on aviation procedures.
Use numbered steps for procedures. Include exact values, speeds, and warnings.

RETRIEVED MANUAL EXCERPTS:
───────────────────────────
{context}
───────────────────────────"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": req.question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources_used": len(relevant_chunks),
        "model": "llama-3.3-70b-versatile (Groq)",
        "chunks_retrieved": [c[:120] + "..." for c in relevant_chunks],
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = sessions[session_id]
    return {"filename": s["filename"], "chunks": s["chunk_count"]}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        chroma_client.delete_collection(session_id)
        del sessions[session_id]
    return {"message": "Session cleared."}


app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/index.html")