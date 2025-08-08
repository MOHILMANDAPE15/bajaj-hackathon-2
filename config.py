import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the LlamaIndex-based system."""
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY_HERE")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "YOUR_PINECONE_ENVIRONMENT_HERE")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policy-index")
    MAX_EMBEDDING_BATCH_SIZE = 100 

    DOCUMENTS_PATH: str = "./data/policies"
    
    LOG_LEVEL: str = "INFO"

    # --- RAG Pipeline Configuration ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    INITIAL_RETRIEVAL_K: int = 7  # Number of docs to fetch initially
    TOP_K_RESULTS: int = 5  # Number of docs to pass to LLM after reranking
    CONFIDENCE_THRESHOLD = 0.4  # Lowered threshold to prevent false negatives
    RERANKER_MODEL: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    # Weights for hybrid search (BM25, Vector Search)
    ENSEMBLE_WEIGHTS: list[float] = [0.5, 0.5]
    
    @classmethod
    def validate(cls) -> None:
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
            raise ValueError("PINECONE_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_ENVIRONMENT or cls.PINECONE_ENVIRONMENT == "YOUR_PINECONE_ENVIRONMENT_HERE":
            raise ValueError("PINECONE_ENVIRONMENT is not set. Please set the environment variable or update the config.py file.")

config = Config()