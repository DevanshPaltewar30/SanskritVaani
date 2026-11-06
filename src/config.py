import os
import psutil

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDING_DIR = os.path.join(MODELS_DIR, "embeddings", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_DIR = os.path.join(MODELS_DIR, "llm")
DEFAULT_LLM_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_LLM_PATH = os.path.join(LLM_DIR, DEFAULT_LLM_FILE)
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# Ensure required directories exist
for d in [DATA_DIR, CORPUS_DIR, UPLOADS_DIR, MODELS_DIR, LLM_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)

# Hardware & CPU Configuration (Strictly CPU Only)
PHYSICAL_CORES = psutil.cpu_count(logical=False) or 4
TOTAL_CORES = psutil.cpu_count(logical=True) or 8
# Allocate conservative thread count to maintain UI responsiveness
DEFAULT_CPU_THREADS = max(2, min(PHYSICAL_CORES, 6))

# RAG & Chunking Parameters
DEFAULT_CHUNK_SIZE = 400  # Sanskrit characters per chunk
DEFAULT_CHUNK_OVERLAP = 80 # Character overlap
DEFAULT_TOP_K = 3         # Retrieved passages count
SCORE_THRESHOLD = 0.38    # Minimum cosine similarity threshold for confident retrieval

# LLM Generation Parameters
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.15 # Low temperature for strict factual grounding
DEFAULT_CONTEXT_WINDOW = 2048

