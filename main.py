#!/usr/bin/env python3
"""
Advanced RAG API Server using LangGraph
This server provides an API for document ingestion and querying using a sophisticated
RAG pipeline built with LangChain and LangGraph.
"""
import logging
import os
import sys

# --- Environment Setup ---
# Explicitly add the virtual environment's site-packages to the system path
# This ensures that all installed packages are discoverable, resolving potential import errors.
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import requests
import tempfile
import uvicorn
import asyncio
import json
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class BatchQueryRequest(BaseModel):
    documents: Optional[str] = Field(None, description="URL to a document to ingest before querying.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Global RAG Service ---
rag_service: Optional[RAGService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    This replaces the deprecated @app.on_event("startup") and "shutdown" decorators.
    """
    global rag_service
    logger.info("🚀 Initializing RAG Service on startup...")
    rag_service = RAGService()
    logger.info("✅ RAG Service initialized.")
    yield
    # Shutdown logic can go here if needed
    logger.info("🔌 System shutting down.")

app = FastAPI(
    title="Advanced RAG API Server",
    description="A sophisticated RAG pipeline using LangGraph for intelligent document querying.",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Verify Bearer token authentication."""
    # This is a placeholder secret token. In a real application, use a secure way to manage this.
    SECRET_TOKEN = "f9e29d7edca43a3e09b4f1c925d7efed93cc349767454bfbb423db67e29741b2"
    if not credentials or credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials

@app.get("/")
async def root():
    return {
        "message": "Advanced RAG API Server",
        "version": app.version,
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Performs a basic health check on the API."""
    logger.info("Performing health check...")
    return {"status": "ok"}

@app.post("/api/v1/hackrx/run", tags=["Hackathon"], response_model=HackRxResponse)
async def process_batch_queries(request: BatchQueryRequest, token: str = Depends(verify_token)):
    """
    Handles document ingestion and batch querying using the advanced RAG service.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Service not available.")

    # --- Step 1: Ingest Document ---
    if not request.documents:
        raise HTTPException(status_code=400, detail="A document URL must be provided.")

    try:
        # Download the document from the URL to a temporary file
        response = requests.get(request.documents, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Document downloaded to {tmp_file_path}. Ingesting...")
        rag_service.ingest_document(tmp_file_path)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    # --- Step 2: Process Questions ---
    logger.info(f"Ingestion complete. Processing {len(request.questions)} questions.")
    
    async def process_single_question(question: str):
        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous graph invocation in a thread pool
            result = await loop.run_in_executor(
                None, rag_service.ask_question, question
            )
            if "error" in result:
                logger.error(f"Error processing question '{question}': {result['error']}")
                return f"Error: {result['error']}"
            
            answer = result.get('answer', 'No answer found.')
            # Ensure the answer is always a string to prevent Pydantic validation errors
            if isinstance(answer, (dict, list)):
                return json.dumps(answer)
            return str(answer)
        except Exception as e:
            logger.error(f"Unhandled exception for question '{question}': {e}", exc_info=True)
            return "An unexpected error occurred."

    tasks = [process_single_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    logger.info("Batch processing complete.")
    return HackRxResponse(answers=answers)
    
if __name__ == "__main__":
    # Use the PORT environment variable if available, otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level=config.LOG_LEVEL.lower()
    )