import os
import urllib.request
import sys
import time

EMBEDDING_DIR = os.path.join("models", "embeddings", "paraphrase-multilingual-MiniLM-L12-v2")
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(os.path.join(EMBEDDING_DIR, "1_Pooling"), exist_ok=True)

BASE_URL = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"

FILES = [
    ("config.json", "config.json"),
    ("config_sentence_transformers.json", "config_sentence_transformers.json"),
    ("modules.json", "modules.json"),
    ("sentence_bert_config.json", "sentence_bert_config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("sentencepiece.bpe.model", "sentencepiece.bpe.model"),
    ("1_Pooling/config.json", "1_Pooling/config.json"),
    ("model.safetensors", "model.safetensors"),
]

def download_file(rel_path, target_path):
    dest = os.path.join(EMBEDDING_DIR, target_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"Already exists: {target_path} ({os.path.getsize(dest):,} bytes)")
        return
    url = f"{BASE_URL}/{rel_path}"
    print(f"Downloading {rel_path} -> {dest} ...")
    t0 = time.time()
    
    # Custom opener with User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "JnanaSetu-Downloader/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB
        last_print = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and time.time() - last_print > 3:
                last_print = time.time()
                pct = (downloaded / total_size) * 100
                print(f"  {target_path}: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
    
    elapsed = time.time() - t0
    print(f"Completed {target_path} in {elapsed:.1f}s ({os.path.getsize(dest):,} bytes)")

def main():
    print("=== Downloading Multilingual MiniLM Embedding Model for CPU ===")
    for rel_path, target_path in FILES:
        download_file(rel_path, target_path)
    print("=== Embedding Model Download Complete! ===")

if __name__ == "__main__":
    main()

