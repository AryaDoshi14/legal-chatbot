"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
Searches across 'offences' (BNS) and 'procedures' (BNSS + remedies).
Reranks using BAAI/bge-reranker-base down to top ~8 chunks.
"""

import os
import sys
import pickle
import chromadb
from typing import List, Dict, Any, Tuple

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer, CrossEncoder
from src.index import CHROMA_DIR, BM25_PATH, EMBEDDING_MODEL_NAME, tokenize_for_bm25

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_CHROMA_CLIENT = None
_OFFENCES_COL = None
_PROCEDURES_COL = None
_BM25_DATA = None
_EMBEDDER = None
_RERANKER = None
_KEY_PROCEDURAL_REMEDIES = None

def get_key_procedural_remedies() -> List[Dict[str, Any]]:
    """Retrieve key statutory procedure chunks (FIR/Zero FIR, refusal to SP, magistrate complaint)."""
    global _KEY_PROCEDURAL_REMEDIES
    if _KEY_PROCEDURAL_REMEDIES is None:
        from src.index import parse_remedies_md
        rems = parse_remedies_md()
        key_chunks = []
        # 1. Zero FIR / FIR: BNSS 173(1)
        c1 = next((r for r in rems if "Zero FIR" in r.get("title", "")), None) or next((r for r in rems if "173(1)" in str(r.get("section", ""))), None)
        if c1:
            key_chunks.append(dict(c1))
        # 2. Refusal to record FIR -> SP: BNSS 173(4)
        c2 = next((r for r in rems if "173(4)" in str(r.get("section", ""))), None)
        if c2:
            key_chunks.append(dict(c2))
        # 3. Magistrate order of investigation: BNSS 175(3)
        c3 = next((r for r in rems if "175(3)" in str(r.get("section", ""))), None)
        if c3:
            key_chunks.append(dict(c3))
        # 4. Private complaint: BNSS 223
        c4 = next((r for r in rems if "223" in str(r.get("section", ""))), None)
        if c4:
            key_chunks.append(dict(c4))
        _KEY_PROCEDURAL_REMEDIES = key_chunks
    return _KEY_PROCEDURAL_REMEDIES

def get_resources():
    global _CHROMA_CLIENT, _OFFENCES_COL, _PROCEDURES_COL, _BM25_DATA, _EMBEDDER, _RERANKER
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
        _OFFENCES_COL = _CHROMA_CLIENT.get_collection(name="offences")
        _PROCEDURES_COL = _CHROMA_CLIENT.get_collection(name="procedures")
        
    if _BM25_DATA is None and os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            _BM25_DATA = pickle.load(f)
            
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
        
    return _CHROMA_CLIENT, _OFFENCES_COL, _PROCEDURES_COL, _BM25_DATA, _EMBEDDER, _RERANKER

def dense_search(query: str, collection, top_k: int = 15) -> List[Dict[str, Any]]:
    """Perform dense vector search on Chroma collection."""
    _, _, _, _, embedder, _ = get_resources()
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    
    hits = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            doc_text = results["documents"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0.0
            hits.append({
                "chunk_id": chunk_id,
                "act": meta.get("act", ""),
                "section": meta.get("section", ""),
                "title": meta.get("title", ""),
                "chapter": meta.get("chapter", ""),
                "type": meta.get("type", ""),
                "source_file": meta.get("source_file", ""),
                "text": doc_text,
                "dense_score": 1.0 - dist
            })
    return hits

def bm25_search(query: str, corpus_key: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """Perform BM25 keyword search on offences or procedures."""
    _, _, _, bm25_data, _, _ = get_resources()
    if not bm25_data or corpus_key not in bm25_data:
        return []
        
    bm25_obj = bm25_data[corpus_key]["bm25"]
    chunks = bm25_data[corpus_key]["chunks"]
    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []
        
    scores = bm25_obj.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    hits = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        c = chunks[idx]
        hits.append({
            "chunk_id": c.get("chunk_id", f"{c['act']}_{c['section']}"),
            "act": c.get("act", ""),
            "section": c.get("section", ""),
            "title": c.get("title", ""),
            "chapter": c.get("chapter", ""),
            "type": c.get("type", ""),
            "source_file": c.get("source_file", ""),
            "text": f"Act: {c['act']} Section: {c['section']} Title: {c['title']}\n{c['text']}",
            "bm25_score": float(scores[idx])
        })
    return hits

def rrf_merge(rank_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
    RRF score = sum(1.0 / (k + rank))
    """
    rrf_scores = {}
    doc_lookup = {}
    
    for r_list in rank_lists:
        for rank, item in enumerate(r_list):
            doc_id = item["chunk_id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_lookup[doc_id] = item
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    merged = []
    for doc_id in sorted_doc_ids:
        doc = dict(doc_lookup[doc_id])
        doc["rrf_score"] = rrf_scores[doc_id]
        merged.append(doc)
    return merged

def rerank_chunks(query: str, candidates: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
    """Rerank candidates using BAAI/bge-reranker-base."""
    if not candidates:
        return []
        
    _, _, _, _, _, reranker = get_resources()
    pairs = [(query, c["text"][:1200]) for c in candidates]
    scores = reranker.predict(pairs)
    
    ranked = []
    for i, c in enumerate(candidates):
        item = dict(c)
        item["rerank_score"] = float(scores[i])
        ranked.append(item)
        
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]

def retrieve(search_queries: List[str], user_situation: str = "", top_k_rerank: int = 8) -> List[Dict[str, Any]]:
    """
    Complete hybrid retrieval pipeline:
    1. For each query, retrieve dense and BM25 hits from 'offences' and 'procedures'.
    2. Merge separately via RRF to guarantee strong candidates for both offences and procedure remedies.
    3. Rerank using bge-reranker-base with full title and statutory context.
    4. Return top reranked chunks (top offences in top slots, followed by top procedural remedies).
    """
    _, offences_col, procedures_col, _, _, rr = get_resources()
    
    all_off_lists = []
    all_proc_lists = []
    
    for q in search_queries:
        d_o = dense_search(q, offences_col, top_k=15)
        b_o = bm25_search(q, "offences", top_k=15)
        d_p = dense_search(q, procedures_col, top_k=15)
        b_p = bm25_search(q, "procedures", top_k=15)
        
        if d_o:
            all_off_lists.append(d_o)
        if b_o:
            all_off_lists.append(b_o)
        if d_p:
            all_proc_lists.append(d_p)
        if b_p:
            all_proc_lists.append(b_p)
            
    # Also search with user_situation if provided
    if user_situation:
        d_o_s = dense_search(user_situation, offences_col, top_k=15)
        d_p_s = dense_search(user_situation, procedures_col, top_k=15)
        if d_o_s:
            all_off_lists.append(d_o_s)
        if d_p_s:
            all_proc_lists.append(d_p_s)
            
    if not all_off_lists and not all_proc_lists:
        return []
        
    merged_off = rrf_merge(all_off_lists, k=60)[:25] if all_off_lists else []
    merged_proc = rrf_merge(all_proc_lists, k=60)[:25] if all_proc_lists else []
    
    # Prioritize practical remedies in procedures
    remedy_procs = [c for c in merged_proc if "remedy" in c.get("source_file", "").lower() or "remedy" in c.get("title", "").lower()]
    other_procs = [c for c in merged_proc if c not in remedy_procs]
    ordered_proc = (remedy_procs + other_procs)[:20]
    
    # Reranking query: combine situation with statutory keywords for high-precision cross-encoding
    key_terms = " ".join([q for q in search_queries if any(k in q for k in ["BNS", "BNSS", "theft", "snatch", "cheat", "murder", "robbery", "hurt", "extort", "trust", "deposit", "trespass", "defamation"])][:2])
    rerank_q = f"{user_situation} {key_terms}".strip() if user_situation else (search_queries[0] if search_queries else "")
    
    # 1. Rerank Offences
    top_off = []
    if merged_off:
        pairs_off = [(rerank_q, f"Act: {c['act']} Section: {c['section']} Title: {c['title']}\n{c['text'][:1200]}") for c in merged_off]
        scores_off = rr.predict(pairs_off)
        for i, c in enumerate(merged_off):
            item = dict(c)
            item["rerank_score"] = float(scores_off[i])
            top_off.append(item)
        top_off.sort(key=lambda x: x["rerank_score"], reverse=True)
        
    # 2. Rerank Procedures (prioritize practical victim remedies like FIR, Zero FIR, refusal)
    top_proc = []
    if ordered_proc:
        pairs_proc = [(user_situation or rerank_q, f"Act: {c['act']} Section: {c['section']} Title: {c['title']}\n{c['text'][:1200]}") for c in ordered_proc]
        scores_proc = rr.predict(pairs_proc)
        for i, c in enumerate(ordered_proc):
            item = dict(c)
            is_remedy = 1 if ("remedy" in c.get("source_file", "").lower() or any(s in str(c.get("section", "")) for s in ["173", "174", "175", "223"])) else 0
            item["rerank_score"] = float(scores_proc[i]) + (0.015 * is_remedy)
            top_proc.append(item)
        top_proc.sort(key=lambda x: x["rerank_score"], reverse=True)
        
    # Balanced combination: Top 4 offences in slots 1-4 (for top-5 relevance),
    # followed by key practical victim procedural remedies (FIR/Zero FIR, refusal to SP, magistrate order)
    if user_situation:
        key_remedies = get_key_procedural_remedies()
        # Merge key remedies with retrieved top_proc without duplicates
        proc_combined = list(key_remedies)
        seen_p = {c.get("chunk_id") for c in proc_combined}
        for p in top_proc:
            if p.get("chunk_id") not in seen_p:
                proc_combined.append(p)
                seen_p.add(p.get("chunk_id"))
        final_chunks = top_off[:4] + proc_combined[:4]
    else:
        final_chunks = top_off[:4] + top_proc[:4]

    if len(final_chunks) < top_k_rerank:
        remaining_off = [c for c in top_off if c not in final_chunks]
        final_chunks.extend(remaining_off[:(top_k_rerank - len(final_chunks))])
    if len(final_chunks) < top_k_rerank:
        remaining_proc = [c for c in top_proc if c not in final_chunks]
        final_chunks.extend(remaining_proc[:(top_k_rerank - len(final_chunks))])
        
    return final_chunks[:top_k_rerank]

if __name__ == "__main__":
    print("Testing Hybrid Retrieval & Reranker...")
    test_queries = [
        "theft movable property dishonestly BNS 303",
        "purse snatching BNS 304",
        "FIR registration procedure BNSS 173"
    ]
    situation = "two boys on a motorcycle snatched my gold chain while I was walking home"
    results = retrieve(test_queries, user_situation=situation, top_k_rerank=8)
    print(f"\nTop {len(results)} retrieved chunks:")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['act']} Sec {r['section']} ({r['type']}): {r['title']} | Rerank Score: {r['rerank_score']:.4f}")
