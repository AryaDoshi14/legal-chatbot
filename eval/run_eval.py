"""
Automated evaluation runner for Legal RAG Chatbot.
Evaluates 40 test cases:
- Retrieval Hit Rate @ Top 8: Expected section present in retrieved chunks
- Answer Citation Accuracy: Expected section cited in verified output
Saves comprehensive results to eval/results.md.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyze import analyze_query
from src.retrieve import retrieve
from src.answer import generate_answer
from src.verify import verify_citations, normalize_section_str

TEST_CASES_PATH = "eval/test_cases.json"
RESULTS_PATH = "eval/results.md"

require_llm = "--require-llm" in sys.argv

def run_evaluation():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    print(f"Starting evaluation on {len(test_cases)} test cases...")
    
    results = []
    retrieval_hits = 0
    citation_hits = 0
    start_time = time.time()
    
    for tc in test_cases:
        tc_id = tc["id"]
        query = tc["query"]
        expected_sections = [normalize_section_str(s) for s in tc["expected_bns_sections"]]
        tc_type = tc["type"]
        
        # 1. Analyze
        analysis = analyze_query(query)
        queries = analysis.get("search_queries", [query])
        
        # 2. Retrieve
        retrieved = retrieve(queries, user_situation=query, top_k_rerank=8)
        retrieved_secs = [normalize_section_str(c.get("section", "")) for c in retrieved]
        
        # Check retrieval hit
        has_retrieval_hit = any(exp in retrieved_secs for exp in expected_sections)
        if has_retrieval_hit:
            retrieval_hits += 1
            
        # 3. Answer
        answer = generate_answer(query, analysis, retrieved)
        if require_llm and answer.get("is_fallback"):
            print("\n" + "!"*80)
            print(f"ABORTED on test case {tc_id}: --require-llm was specified, but answer used fallback mode!")
            print(f"Reason: {answer.get('fallback_reason', 'unknown')}")
            print("!"*80)
            sys.exit(1)
        
        # 4. Verify
        verified = verify_citations(answer, retrieved)
        cited_secs = [normalize_section_str(s.get("section", "")) for s in verified.get("applicable_sections", [])]
        
        # Check citation hit
        has_citation_hit = any(exp in cited_secs for exp in expected_sections)
        if has_citation_hit:
            citation_hits += 1
            
        res_record = {
            "id": tc_id,
            "type": tc_type,
            "query": query,
            "expected": expected_sections,
            "retrieved_top8": retrieved_secs,
            "cited_sections": cited_secs,
            "retrieval_hit": has_retrieval_hit,
            "citation_hit": has_citation_hit
        }
        results.append(res_record)
        
        status = "HIT" if has_retrieval_hit else "MISS"
        cite_status = "CITED" if has_citation_hit else "UNCITED"
        print(f"[{tc_id:02d}/40] {tc_type.upper()}: {query[:45]}... -> Retrieval: {status} | Citation: {cite_status}")
        
    total = len(test_cases)
    retrieval_rate = (retrieval_hits / total) * 100.0
    citation_rate = (citation_hits / total) * 100.0
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Test Cases:       {total}")
    print(f"Retrieval Hit Rate @ 8: {retrieval_hits}/{total} ({retrieval_rate:.1f}%)")
    print(f"Answer Citation Acc:    {citation_hits}/{total} ({citation_rate:.1f}%)")
    print(f"Total Elapsed Time:     {elapsed:.1f}s")
    print("="*50)
    
    # Write eval/results.md
    md_content = [
        "# Evaluation Benchmark Results",
        "",
        "## Summary Metrics",
        f"- **Total Test Cases**: {total} (12 Direct Questions, 28 Situations)",
        f"- **Retrieval Hit Rate (Top-8)**: **{retrieval_rate:.1f}%** ({retrieval_hits}/{total})",
        f"- **Answer Citation Accuracy**: **{citation_rate:.1f}%** ({citation_hits}/{total})",
        f"- **Evaluation Duration**: {elapsed:.1f} seconds",
        "",
        "## Methodology",
        "1. **Analyze**: Query analyzer extracts candidate offences, legal queries, and resolves IPC/BNS ambiguity.",
        "2. **Retrieve**: Hybrid multi-query retrieval (dense `bge-base-en-v1.5` + BM25) with Reciprocal Rank Fusion over BNS and BNSS chunks, reranked via `bge-reranker-base` to top 8.",
        "3. **Answer**: Grounded generation strictly from retrieved chunks.",
        "4. **Verify**: Deterministic code verification ensuring all cited sections exist in retrieved context.",
        "",
        "## Detailed Case Results",
        "| ID | Type | Query | Expected Sections | Retrieved Top-8 Sections | Retrieval Hit | Citation Hit |",
        "|---|---|---|---|---|:---:|:---:|"
    ]
    
    for r in results:
        exp_str = ", ".join(r["expected"])
        ret_str = ", ".join(r["retrieved_top8"][:6])
        r_hit = "✅" if r["retrieval_hit"] else "❌"
        c_hit = "✅" if r["citation_hit"] else "❌"
        q_escaped = r["query"].replace("|", "\\|")
        md_content.append(f"| {r['id']} | {r['type']} | {q_escaped} | {exp_str} | {ret_str} | {r_hit} | {c_hit} |")
        
    md_content.append("")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"Results written to {RESULTS_PATH}")
    return retrieval_rate, citation_rate

if __name__ == "__main__":
    run_evaluation()

