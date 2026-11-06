#!/usr/bin/env python3
"""Generate a professional PDF technical report for the JnanaSetu project.
Uses the actual evaluation results from `evaluation_results.json`.
"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation_results.json")
PDF_PATH = os.path.join(PROJECT_ROOT, "JnanaSetu_Technical_Report.pdf")

# Load evaluation results
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
summary = data.get("summary", {})
per_query = data.get("per_query_details", [])

c = canvas.Canvas(PDF_PATH, pagesize=A4)
width, height = A4
margin = 50
y = height - margin

def draw_header(text, size=16, leading=20):
    global y
    c.setFont("Helvetica-Bold", size)
    c.drawString(margin, y, text)
    y -= leading

def draw_para(text, size=10, leading=14):
    global y
    c.setFont("Helvetica", size)
    for line in text.split('\n'):
        c.drawString(margin, y, line)
        y -= leading
    y -= leading/2

def new_page():
    global y
    c.showPage()
    y = height - margin

# Title page
draw_header("JnanaSetu (ज्ञानसेतुः) – Sanskrit Knowledge Assistant", size=20, leading=30)
draw_para(f"Technical Report – {datetime.now().strftime('%Y-%m-%d')}", size=12, leading=18)
new_page()

# 1. Project Overview
draw_header("1. Project Overview")
overview = (
    "JnanaSetu is a locally‑run Retrieval‑Augmented Generation (RAG) system that answers "
    "questions over a curated corpus of five classical Sanskrit teaching stories. "
    "It supports queries in Devanagari, English, and Roman‑transliterated Sanskrit, "
    "operating strictly on CPU without any cloud services."
)
draw_para(overview)

# 2. System Architecture
new_page()
draw_header("2. System Architecture")
arch = (
    "The architecture consists of four main components: \n"
    "• Sanskrit NLP pre‑processor – script detection, normalization, and transliteration.\n"
    "• Multilingual Sentence‑Transformer embeddings (CPU‑only) that embed both queries and document chunks.\n"
    "• FAISS Index (Inner‑Product) storing 94 vectors for the corpus plus uploaded documents.\n"
    "• Local LLM (GGUF format) loaded via llama.cpp with n_gpu_layers=0.\n"
    "The retrieval pipeline first checks the similarity score against a confidence threshold; "
    "if below, a grounded refusal response is returned."
)
draw_para(arch)

# 3. Data & Pre‑processing
new_page()
draw_header("3. Data & Pre‑processing")
data_desc = (
    "The corpus contains five stories provided as PDF and TXT files in `data/corpus/`. "
    "Each document is loaded by `SanskritDocumentLoader`, normalized to canonical Devanagari, "
    "and split into overlapping chunks (150 tokens, 30 token overlap) preserving punctuation."
)
draw_para(data_desc)

# 4. Embeddings & Retrieval
new_page()
draw_header("4. Embeddings & Retrieval")
emb_desc = (
    "Embeddings use the `paraphrase‑multilingual‑MiniLM‑L12‑v2` model from Sentence‑Transformers, "
    "configured for CPU (`device='cpu'`). Chunk vectors are stored in a FAISS IndexFlatIP, "
    "which implements cosine similarity via inner‑product on L2‑normalized vectors. "
    "During query time, the system adds a small token‑overlap boost and a source‑bias boost to "
    "prefer corpus documents over generic uploads."
)
draw_para(emb_desc)

# 5. CPU‑Only LLM Inference
new_page()
draw_header("5. CPU‑Only LLM Inference")
llm_desc = (
    "The answer generation stage uses a 0.5B parameter GGUF model (`qwen2.5-0.5b‑instruct-q4_k_m.gguf`) "
    "loaded through `llama_cpp` with `n_gpu_layers=0`. All inference runs on CPU across 6 threads, "
    "ensuring the project meets the strict hardware compliance requirement."
)
draw_para(llm_desc)

# 6. Evaluation Metrics (Actual Results)
new_page()
draw_header("6. Evaluation Metrics (Actual Results)")
metrics = (
    f"Mean Reciprocal Rank (MRR): {summary.get('mrr', 0):.4f}\n"
    f"Precision@3: {summary.get('precision_at_k', 0):.4f}\n"
    f"Recall@3: {summary.get('recall_at_k', 0):.4f}\n"
    f"Answer Faithfulness: {summary.get('faithfulness', 0):.4f}\n"
    f"Answer Relevance: {summary.get('answer_relevance', 0):.4f}\n"
    f"Avg Retrieval Latency: {summary.get('avg_retrieval_time_ms', 0):.2f} ms\n"
    f"Avg Generation Time: {summary.get('avg_generation_time_s', 0):.2f} s\n"
    f"Avg Tokens / Second: {summary.get('avg_tokens_per_second', 0):.1f} tps\n"
    f"Avg Embedding Time: {summary.get('avg_embedding_time_ms', 0):.2f} ms\n"
    f"Avg CPU Utilization: {summary.get('avg_cpu_utilization_percent', 0):.1f}%\n"
    f"Avg RAM Consumption: {summary.get('avg_ram_consumption_mb', 0):.1f} MB"
)
draw_para(metrics)

# 7. Limitations & Future Work
new_page()
draw_header("7. Limitations & Future Work")
limits = (
    "* Retrieval confidence still depends on the similarity threshold; some out‑of‑domain queries may retrieve "
    "the generic `Rag‑docs.pdf`.\n"
    "* The GGUF model is small (0.5 B) and produces slower response times (~9 s). Larger CPU‑efficient models could improve speed.\n"
    "* The current corpus covers only five stories; scaling to larger Sanskrit corpora will require re‑indexing and possibly dimensionality reduction."
)
draw_para(limits)

c.save()
print(f"Report written to {PDF_PATH}")

