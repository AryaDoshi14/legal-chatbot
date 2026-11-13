# Legal RAG Chatbot: Indian Criminal Law (BNS + BNSS)

## Goal
User describes a situation or asks a direct question. The bot says which BNS
section(s) may apply and why, and what the person can do (BNSS procedure).
Everything comes only from retrieved law text.

## Input files (in ./data/, use the actual names found there)
BNS.pdf, BNSS.pdf, IPC_BNS_mapping.pdf, BNS_summary.pdf (optional,
background only, do NOT index).

## Pipeline
1. ANALYZE (LLM): message -> JSON {mode: question|situation, facts,
   candidate_offences, search_queries (3-5, legal wording), missing_facts}.
   Before the LLM call, detect section numbers the user mentions.
   Ambiguity rule: a bare number (e.g. "302") is ambiguous. Treat it as IPC
   only if the user says IPC, as BNS only if the user says BNS, otherwise
   show both mappings.
2. RETRIEVE: per query, BM25 + dense search (bge-base-en-v1.5) over
   "offences" (BNS) and "procedures" (BNSS + remedies). Merge with
   reciprocal rank fusion. Rerank with bge-reranker-base to top ~8.
3. ANSWER (LLM, claude-sonnet-4-6): use ONLY retrieved chunks. JSON output:
   applicable_sections (act, section, title, why_it_applies, punishment),
   missing_facts_and_how_section_changes, what_you_can_do, caveats.
   If the chunks do not support an answer, reply "not found in the loaded
   law text". Use "may fall under", never a verdict.
4. VERIFY (plain code): every cited section must exist in the retrieved
   chunks. Remove or flag any that do not.

## Data rules
- One section = one chunk. Keep illustrations, explanations, exceptions and
  sub-sections inside their section chunk. Split only very long sections,
  keeping act/section/title in every piece.
- Metadata: act, section, title, chapter, type, source_file.
- Skip front matter, indexes, Hindi text, headers, footers.
- IPC->BNS mapping saved to data/ipc_bns_mapping.json.
- data/remedies.md: 25-30 short chunks (FIR, zero FIR, police refusal,
  complaint to Magistrate, medical exam, victim rights, etc.), each tied to
  a BNSS section that was LOOKED UP in the parsed BNSS. Never from memory.

## Stack (do not substitute without asking)
Python, PyMuPDF/pdfplumber, sentence-transformers, Chroma (persistent),
rank_bm25, bge-reranker-base, Anthropic API, Streamlit. API key only from
the ANTHROPIC_API_KEY environment variable.

## UI
Streamlit chat: section cards (act, section, title, why it applies,
punishment, expandable source text), "what you can do" steps, "details that
would change the answer", and a visible disclaimer (general information,
not legal advice).

## Evaluation
eval/test_cases.json: 40 cases (12 direct questions, 28 situations,
including ambiguous ones and some using old IPC numbers). Report retrieval
hit rate (expected section in top 8) and citation accuracy. Run once, fix
only obvious bugs, re-run at most twice.

## Deliverables
data/, src/ (ingest, retrieve, analyze, answer, verify), app.py, eval/,
requirements.txt, .gitignore, README.md (Mermaid architecture diagram,
setup, how to run, eval results, limitations, disclaimer).
