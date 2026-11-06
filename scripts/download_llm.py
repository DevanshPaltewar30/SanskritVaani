import os
import urllib.request
import time
import sys

MODEL_DIR = os.path.join("models", "llm")
os.makedirs(MODEL_DIR, exist_ok=True)

# Qwen2.5-0.5B-Instruct-GGUF is ultra-lightweight (397 MB), extremely fast on CPU, and supports multilingual/Indic
DEFAULT_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/{DEFAULT_MODEL}"

def download_llm(filename=DEFAULT_MODEL, url=MODEL_URL):
    dest = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
        print(f"LLM model already exists: {dest} ({os.path.getsize(dest):,} bytes)")
        return dest
        
    print(f"Downloading GGUF LLM: {filename} from {url} ...")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "JnanaSetu-Downloader/1.0"})
    
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2MB
        last_print = 0
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and time.time() - last_print > 4:
                last_print = time.time()
                pct = (downloaded / total_size) * 100
                speed = (downloaded / (time.time() - t0)) / (1024 * 1024)
                print(f"  {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) - {speed:.1f} MB/s")
                
    elapsed = time.time() - t0
    print(f"Successfully downloaded {filename} in {elapsed:.1f}s ({os.path.getsize(dest):,} bytes)!")
    return dest

if __name__ == "__main__":
    download_llm()

