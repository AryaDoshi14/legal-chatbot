"""
Deterministic, code-based citation verification.
Ensures every cited section exists in the retrieved chunks.
Enforces relevance check: cited applicable sections MUST be in the top 5 reranked chunks.
Prevents cross-act collisions: every lookup key is strictly (act, section).
Flags or prunes ungrounded citations and attaches expandable source text for the UI.
"""

import re
from typing import Dict, Any, List

def normalize_section_str(sec: str) -> str:
    """Normalize section string for robust matching."""
    s = str(sec).strip().lower()
    # Strip leading 'section' or 'sec'
    s = re.sub(r"^(?:section|sec\.?)\s*", "", s)
    # Extract base number + optional subclause
    m = re.match(r"^(\d+[a-z]?)", s)
    return m.group(1) if m else s

def verify_citations(answer: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate all citations in the generated answer against the retrieved chunks.
    No LLM is used; this is purely deterministic verification code.
    
    Relevance check: Cited applicable sections MUST appear in the top 5 reranked chunks.
    Collision prevention: Strictly keys lookups by (Act, Section).
    """
    verified_answer = dict(answer)
    
    # ── 1. Build strict (act, section) maps ──
    top5_chunks = retrieved_chunks[:5]
    top5_map = {}
    for c in top5_chunks:
        act = c.get("act", "").upper()
        raw_sec = str(c.get("section", "")).strip()
        norm_sec = normalize_section_str(raw_sec)
        top5_map[(act, norm_sec)] = c
        top5_map[(act, raw_sec.lower())] = c
        if c.get("chunk_id"):
            top5_map[c["chunk_id"].upper()] = c
            
    all_chunks_map = {}
    for c in retrieved_chunks:
        act = c.get("act", "").upper()
        raw_sec = str(c.get("section", "")).strip()
        norm_sec = normalize_section_str(raw_sec)
        all_chunks_map[(act, norm_sec)] = c
        all_chunks_map[(act, raw_sec.lower())] = c
        if c.get("chunk_id"):
            all_chunks_map[c["chunk_id"].upper()] = c
            
    verified_applicable = []
    unverified_citations = []
    
    # ── 2. Verify applicable sections (MUST be in top 5, strictly matching Act + Section) ──
    for item in answer.get("applicable_sections", []):
        item_act = item.get("act", "BNS").upper()
        raw_sec = str(item.get("section", "")).strip()
        norm_sec = normalize_section_str(raw_sec)
        
        # Strict (item_act, norm_sec) check — NEVER cross-match against another Act
        matched_chunk = top5_map.get((item_act, norm_sec)) or top5_map.get((item_act, raw_sec.lower()))
        
        if matched_chunk:
            item_copy = dict(item)
            item_copy["act"] = matched_chunk.get("act", item_act)
            item_copy["verified"] = True
            item_copy["in_top_5"] = True
            item_copy["source_file"] = matched_chunk.get("source_file", "")
            item_copy["source_title"] = matched_chunk.get("title", "")
            item_copy["source_chapter"] = matched_chunk.get("chapter", "")
            item_copy["source_text"] = matched_chunk.get("text", "")
            verified_applicable.append(item_copy)
        else:
            unverified_citations.append(f"{item_act} Section {raw_sec} (not in top 5 reranked chunks)")
            
    verified_answer["applicable_sections"] = verified_applicable
    
    # ── 3. Verify ambiguous groups if present ──
    verified_ambiguous = []
    for grp in answer.get("ambiguous_groups", []):
        grp_copy = dict(grp)
        grp_act = "BNSS" if "BNSS" in grp.get("mapped_bns_section", "").upper() else "BNS"
        m = re.search(r"\b\d+[A-Za-z]?\b", grp.get("mapped_bns_section", ""))
        if m:
            ns = normalize_section_str(m.group(0))
            matched = top5_map.get((grp_act, ns)) or all_chunks_map.get((grp_act, ns))
            if matched:
                grp_copy["verified"] = True
                grp_copy["source_text"] = matched.get("text", "")
        verified_ambiguous.append(grp_copy)
    verified_answer["ambiguous_groups"] = verified_ambiguous

    # ── 4. Verify procedure what_you_can_do steps ──
    verified_steps = []
    for step in answer.get("what_you_can_do", []):
        step_copy = dict(step)
        basis = step.get("section_basis", "")
        
        # Determine Act from basis: check BNSS first, then BNS, default to BNSS
        if re.search(r"\bBNSS\b", basis, re.IGNORECASE):
            step_act = "BNSS"
        elif re.search(r"\bBNS\b", basis, re.IGNORECASE):
            step_act = "BNS"
        else:
            step_act = "BNSS"
            
        sec_nums = re.findall(r"\b\d+[A-Za-z]?(?:\(\d+\))?", basis)
        matched_chunk = None
        for sn in sec_nums:
            nsn = normalize_section_str(sn)
            # Strict lookup under (step_act, nsn)
            matched_chunk = all_chunks_map.get((step_act, nsn)) or all_chunks_map.get((step_act, sn.lower()))
            if matched_chunk:
                break
                
        if matched_chunk:
            step_copy["verified"] = True
            step_copy["source_act"] = matched_chunk.get("act", step_act)
            step_copy["source_section"] = matched_chunk.get("section", "")
            step_copy["source_text"] = matched_chunk.get("text", "")
            verified_steps.append(step_copy)
        else:
            # Drop any step whose BNSS section was not retrieved
            unverified_citations.append(f"Procedural Step '{step.get('action', '')[:35]}...' dropped: {basis} not in retrieved chunks")
            
    # Renumber remaining verified steps
    for idx, stp in enumerate(verified_steps, 1):
        stp["step"] = idx
        
    verified_answer["what_you_can_do"] = verified_steps
    
    # ── 5. Add verification notes to caveats ──
    if "caveats" not in verified_answer or not isinstance(verified_answer["caveats"], list):
        verified_answer["caveats"] = []
        
    if unverified_citations:
        warning_msg = f"Relevance / Grounding Check: The following sections were cited but NOT confirmed within the top 5 reranked statutory chunks and were automatically removed: {', '.join(unverified_citations)}."
        verified_answer["caveats"].insert(0, warning_msg)
        
    verified_answer["verification_summary"] = {
        "total_cited": len(answer.get("applicable_sections", [])),
        "verified_count": len(verified_applicable),
        "unverified_citations": unverified_citations,
        "is_all_verified": len(unverified_citations) == 0
    }
    
    return verified_answer
