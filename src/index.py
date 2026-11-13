"""
Index builder for offences (BNS) and procedures (BNSS + remedies).
Builds persistent local ChromaDB collections with BAAI/bge-base-en-v1.5
and BM25 keyword search indices with rank_bm25.
"""

import os
import re
import json
import pickle
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CHROMA_DIR = "./chroma_db"
BM25_PATH = "./data/bm25_indices.pkl"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

def parse_remedies_md(path: str = "data/remedies.md") -> List[Dict[str, Any]]:
    """Parse handwritten remedies into structured chunks."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    chunks = []
    # Split by ### headings
    items = re.split(r"\n###\s+", content)
    for idx, item in enumerate(items[1:], 1):
        lines = item.strip().split("\n")
        header = lines[0].strip()
        header_clean = re.sub(r"^\d+\.\s*", "", header)
        
        body = "\n".join(lines[1:]).strip()
        # Extract metadata
        act_sec = re.search(r"\*\*(?:Act & Section|Section)\*\*:\s*([^\n]+)", body)
        title_m = re.search(r"\*\*Title\*\*:\s*([^\n]+)", body)
        
        act = "BNSS"
        section = ""
        if act_sec:
            raw_sec = act_sec.group(1).strip()
            sec_num_m = re.search(r"\b\d+[A-Za-z]?(?:\(\d+\))?", raw_sec)
            section = sec_num_m.group(0) if sec_num_m else raw_sec
        else:
            section = f"remedy_{idx}"
            
        title = title_m.group(1).strip() if title_m else header_clean
        
        chunk = {
            "chunk_id": f"REMEDY_{idx}_{section}",
            "act": act,
            "section": section,
            "title": f"Remedy: {header_clean} ({title})",
            "chapter": "PROCEDURAL REMEDIES",
            "type": "procedure",
            "source_file": "data/remedies.md",
            "page_start": 1,
            "text": f"Remedy: {header_clean}\nAct & Section: BNSS {section}\nTitle: {title}\n{body}"
        }
        chunks.append(chunk)
    return chunks

def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase word tokenizer for BM25 matching."""
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9]+\b", text)]

def build_indices(force_rebuild: bool = False):
    """Build or update ChromaDB and BM25 indices."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Loading chunks from disk...")
    with open("data/bns_chunks.json", "r", encoding="utf-8") as f:
        offences_chunks = json.load(f)
        
    with open("data/bnss_chunks.json", "r", encoding="utf-8") as f:
        bnss_chunks = json.load(f)
        
    remedy_chunks = parse_remedies_md("data/remedies.md")
    print(f"Loaded {len(remedy_chunks)} remedy chunks.")
    
    procedures_chunks = bnss_chunks + remedy_chunks
    print(f"Total offences chunks: {len(offences_chunks)}")
    print(f"Total procedures chunks: {len(procedures_chunks)}")
    
    # 1. BM25 Indices
    print("\nBuilding BM25 indices...")
    offences_corpus = [tokenize_for_bm25(f"{c['act']} {c['section']} {c['title']} {c['text']}") for c in offences_chunks]
    procedures_corpus = [tokenize_for_bm25(f"{c['act']} {c['section']} {c['title']} {c['text']}") for c in procedures_chunks]
    
    offences_bm25 = BM25Okapi(offences_corpus)
    procedures_bm25 = BM25Okapi(procedures_corpus)
    
    bm25_data = {
        "offences": {
            "bm25": offences_bm25,
            "chunks": offences_chunks
        },
        "procedures": {
            "bm25": procedures_bm25,
            "chunks": procedures_chunks
        }
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"Saved BM25 indices to {BM25_PATH}")
    
    # 2. ChromaDB Embeddings
    print(f"\nInitializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    if force_rebuild:
        for c_name in ["offences", "procedures"]:
            try:
                client.delete_collection(c_name)
            except Exception:
                pass
                
    offences_col = client.get_or_create_collection(name="offences")
    procedures_col = client.get_or_create_collection(name="procedures")
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Index offences if empty or forced
    if offences_col.count() < len(offences_chunks) or force_rebuild:
        print(f"Generating embeddings for {len(offences_chunks)} offences chunks...")
        batch_size = 64
        for i in range(0, len(offences_chunks), batch_size):
            batch = offences_chunks[i:i + batch_size]
            texts = [f"Act: {c['act']} Section: {c['section']} Title: {c['title']}\n{c['text']}" for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [{
                "act": str(c["act"]),
                "section": str(c["section"]),
                "title": str(c["title"]),
                "chapter": str(c.get("chapter", "")),
                "type": str(c["type"]),
                "source_file": str(c["source_file"])
            } for c in batch]
            
            embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
            offences_col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            print(f"Indexed offences: {min(i + batch_size, len(offences_chunks))}/{len(offences_chunks)}")
    else:
        print(f"Offences collection already populated with {offences_col.count()} documents.")
        
    # Index procedures if empty or forced
    if procedures_col.count() < len(procedures_chunks) or force_rebuild:
        print(f"Generating embeddings for {len(procedures_chunks)} procedures chunks...")
        batch_size = 64
        for i in range(0, len(procedures_chunks), batch_size):
            batch = procedures_chunks[i:i + batch_size]
            texts = [f"Act: {c['act']} Section: {c['section']} Title: {c['title']}\n{c['text']}" for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [{
                "act": str(c["act"]),
                "section": str(c["section"]),
                "title": str(c["title"]),
                "chapter": str(c.get("chapter", "")),
                "type": str(c["type"]),
                "source_file": str(c["source_file"])
            } for c in batch]
            
            embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
            procedures_col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            print(f"Indexed procedures: {min(i + batch_size, len(procedures_chunks))}/{len(procedures_chunks)}")
    else:
        print(f"Procedures collection already populated with {procedures_col.count()} documents.")
        
    print("\nIndexing complete!")
    print(f"Chroma offences count: {offences_col.count()}")
    print(f"Chroma procedures count: {procedures_col.count()}")

if __name__ == "__main__":
    build_indices(force_rebuild=True)

