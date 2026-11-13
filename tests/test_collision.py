"""
Unit test for Issue 1: Act + Section Collision Fix
Ensures that BNS 173 (Bribery) and BNSS 173(1) (FIR) never collide,
and that source text lookups and UI keys use strictly (act, section).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verify import verify_citations, normalize_section_str

def test_no_collision_bns_bnss_173():
    bns_173_chunk = {
        "chunk_id": "BNS_173",
        "act": "BNS",
        "section": "173",
        "title": "Threat Of Injury To Induce Person To Refrain From Applying For Protection To Public Servant",
        "type": "offence",
        "source_file": "data/bns.pdf",
        "text": "173. Whoever holds out any threat of injury to any person for the purpose of inducing that person to refrain or desist from making a legal application for protection against any injury to any public servant legally empowered as such to give such protection..."
    }
    
    bnss_173_chunk = {
        "chunk_id": "BNSS_173_1",
        "act": "BNSS",
        "section": "173(1)",
        "title": "Information in cognizable cases (FIR)",
        "type": "procedure",
        "source_file": "data/bnss.pdf",
        "text": "173. (1) Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, and be read over to the informant..."
    }
    
    # Both chunks present in retrieved list
    retrieved = [bns_173_chunk, bnss_173_chunk]
    
    answer = {
        "applicable_sections": [
            {
                "act": "BNS",
                "section": "173",
                "title": "Threat of injury",
                "why_it_applies": "Threatened to prevent report"
            }
        ],
        "what_you_can_do": [
            {
                "step": 1,
                "action": "File an FIR at police station",
                "section_basis": "BNSS Section 173(1)"
            }
        ]
    }
    
    verified = verify_citations(answer, retrieved)
    
    # 1. Check BNS 173 card text
    bns_sec = verified["applicable_sections"][0]
    assert bns_sec["act"] == "BNS", f"Expected BNS, got {bns_sec['act']}"
    assert "threat of injury" in bns_sec["source_text"].lower(), "BNS 173 card must contain BNS threat of injury text!"
    assert "information relating to the commission" not in bns_sec["source_text"].lower(), "BNS 173 card must NOT contain BNSS FIR text!"
    
    # 2. Check BNSS 173(1) step text
    bnss_step = verified["what_you_can_do"][0]
    assert "information relating to the commission" in bnss_step["source_text"].lower(), "BNSS 173(1) must contain BNSS FIR text!"
    assert "threat of injury" not in bnss_step["source_text"].lower(), "BNSS 173(1) must NOT contain BNS bribery text!"
    
    print("TEST PASSED: BNS 173 and BNSS 173(1) verified with strict (Act, Section) keys without cross-act collision!")

if __name__ == "__main__":
    test_no_collision_bns_bnss_173()

