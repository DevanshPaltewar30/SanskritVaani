import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.engine import JnanaSetuEngine

def main():
    print("Step 1: Initializing JnanaSetuEngine...", flush=True)
    engine = JnanaSetuEngine()
    print("Step 2: Engine initialized successfully!", flush=True)

    questions = [
        "भृत्येन खड्गेन किं कृतम्?",
        "How did the old woman protect her house from thieves in the story?",
        "What is quantum computing?"
    ]

    for q in questions:
        print("\n" + "="*60, flush=True)
        print("User Question:", q, flush=True)
        res = engine.ask(q)
        print("Answer:\n", res["answer"], flush=True)
        print("-" * 40, flush=True)
        print(f"Retrieval Time: {res['retrieval_time']:.3f}s", flush=True)
        print(f"Generation Time: {res['generation_time']:.3f}s", flush=True)
        print(f"Tokens/sec: {res['tokens_per_second']:.1f}", flush=True)
        print(f"Sources returned: {len(res['sources'])}", flush=True)
        if res["sources"]:
            top = res["sources"][0]
            print(f"Top Source: {top['metadata'].get('source')} (Score: {top['score']})", flush=True)

if __name__ == "__main__":
    main()
