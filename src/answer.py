"""
Grounded answer generation using OpenRouter (GPT-4o).
Enforces strict grounding on retrieved chunks, JSON schema output, and safety rules.
Includes Civil-vs-Criminal check, direct-lookup formatting, and plain-language victim remedies.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCLAIMER_TEXT = "Disclaimer: This analysis is for general informational and educational purposes only and does not constitute formal legal advice. Consult a qualified advocate for legal representation."

def generate_answer(user_message: str, analysis: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate structured, grounded answer from retrieved chunks using OpenRouter (GPT-4o).
    Falls back to structured heuristic generator if API key is missing or fails.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        reason = "OPENROUTER_API_KEY not set in environment"
        print(f"[FALLBACK TRIGGERED] {reason}")
        return _heuristic_answer(user_message, analysis, retrieved_chunks, fallback_reason=reason)
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    model_name = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
    print(f"[LLM Answer] Model: {model_name} | Temperature: 0.0")
    
    # Format retrieved chunks into context string with rerank scores
    chunk_contexts = []
    for idx, c in enumerate(retrieved_chunks, 1):
        chunk_contexts.append(
            f"=== CHUNK [{idx}] ===\n"
            f"Act: {c.get('act', '')}\n"
            f"Section: {c.get('section', '')}\n"
            f"Title: {c.get('title', '')}\n"
            f"Type: {c.get('type', '')}\n"
            f"Rerank Score: {c.get('rerank_score', c.get('score', 0.0)):.4f}\n"
            f"Source: {c.get('source_file', '')}\n"
            f"Text:\n{c.get('text', '')}\n"
        )
    retrieved_text = "\n".join(chunk_contexts) if chunk_contexts else "NO CHUNKS RETRIEVED."
    
    system_prompt = f"""You are an elite Indian Criminal Law assistant specializing in the Bharatiya Nyaya Sanhita, 2023 (BNS) and Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS).

CRITICAL STATUTORY GROUNDING & SELECTION RULES:
1. Grounding: Answer ONLY from the retrieved statutory chunks provided below. Never invent sections. If retrieved text contains no matching section, return an empty "applicable_sections" list and state "not found in the loaded law text" in caveats.
2. DECISION RULE — BASE OFFENCE VS. SPECIFIC/AGGRAVATED OFFENCES:
   - Pick the BASE offence that the stated facts fully support.
   - Always show the BASE offence as Rank 1 when it is retrieved in the statutory chunks.
   - A more specific or aggravated section (such as snatching under BNS 304, robbery under BNS 309, theft in dwelling/building under BNS 305, theft by clerk/servant, or theft with deadly weapon) goes in "missing_facts_and_how_section_changes" (or as a conditional alternative at Rank 2/3) UNLESS the user's text explicitly states the specific factual elements it needs.
   - NEVER assume or invent facts not stated by the user.
   - MANDATORY PRINCIPLE & EXAMPLE:
     "my bicycle was stolen from outside my house" -> BNS Section 303 (Theft) is the PRIMARY BASE OFFENCE (Rank 1); snatching (BNS 304) and theft in dwelling (BNS 305) are conditional on facts the user hasn't given (e.g. whether it was suddenly seized from the rider's person with quickness/force, or whether it was inside an enclosed dwelling).
   - Similarly, if the user states that a purse was stolen at a bus stop without stating that it was suddenly grabbed with quickness or physical force from their hands/body, BNS Section 303 (Theft) is the PRIMARY BASE OFFENCE (Rank 1). Snatching (BNS 304) is conditional (Rank 2) requiring proof of sudden quick seizure from the person.
   - BNS Section 305 requires theft in a dwelling house, building, means of transport (inside a vehicle/bus/train), place of worship, or government property. Theft at an open "bus stop" or street does NOT fall under Section 305.
   - BNSS Section 397 provides free first-aid/treatment specifically to victims of acid attack (BNS 124(1)) and sexual offences (BNS 64-71), NOT general fights or altercations.
3. Candidate Offence Ranking:
   - Return at most THREE (3) candidate sections:
     * Rank 1: The BASE offence matching the stated facts.
     * Rank 2 & 3: At most 2 conditional alternatives, each with the exact missing fact it needs specified in "depends_on_missing_fact".
   - Sections about dealing with property afterwards (e.g. BNS 317 Stolen property, BNS 252 Taking gift to recover stolen property, BNS 320 Dishonestly receiving stolen property) must NEVER be Rank 1. Place them under "related_sections" only.
4. Civil-vs-Criminal Check: If the facts describe a contractual, tenancy, business, or money dispute (e.g. landlord withholding security deposit, unpaid loan, service delay) with no clear dishonest inducement from the inception:
   - State clearly that the matter is primarily civil in nature (breach of contract).
   - Name the usual civil remedy (Legal demand notice through an advocate, Summary Suit for recovery under Order XXXVII CPC, or Rent Tribunal / Consumer Forum).
   - Mention criminal sections (BNS 316 Criminal Breach of Trust or BNS 318 Cheating) ONLY as conditional upon proving fraudulent intention from the start or dishonest conversion.
5. Direct Section Lookups:
   - If the user asks a direct question or lookup (e.g., "What is Section 420?", "What is Section 302?", "punishment for cheating?"): Do NOT provide any "what_you_can_do" procedural steps. Set "what_you_can_do" to an empty list [].
   - If a bare section number is ambiguous (e.g., "302"): Do NOT use ranks 1-3. Instead, populate "ambiguous_groups" with two distinct groups:
     (1) "If you meant the old IPC section (IPC Section X)" -> corresponding BNS section and explanation.
     (2) "If you meant Bharatiya Nyaya Sanhita (BNS Section X)" -> BNS section definition and explanation.
6. Procedural Steps ("what_you_can_do"):
   - When applicable (for situations/crimes), you MUST ONLY include procedural steps whose statutory basis (BNSS section) appears in the RETRIEVED STATUTORY CHUNKS below.
   - If a section (e.g., BNSS 173(4) or BNSS 175(3)) was NOT retrieved, DO NOT include that step. Drop any step whose BNSS section was not retrieved.
   - Steps must be written in plain language, in sensible chronological order (FIR / Zero FIR under BNSS 173(1) first, refusal complaint to SP under 173(4) second, Judicial Magistrate under 175(3)/223 third). Never use trial/attachment sections as first steps. Never use "Invoke ... to report or address".
7. Tentative Phrasing: Use tentative language ("may fall under", "appears to attract"), never a conclusive verdict.
8. Output format: Strict JSON only.

REQUIRED JSON SCHEMA:
{{
  "civil_vs_criminal_check": {{
    "is_likely_civil": true or false,
    "assessment": "Explanation if civil/contractual dispute, or null",
    "usual_civil_remedy": "Name of legal notice / CPC Order 37 / Rent Tribunal remedy, or null",
    "criminal_applicability": "Conditional explanation for criminal charges, or null"
  }},
  "applicable_sections": [
    {{
      "rank": 1,
      "act": "BNS",
      "section": "exact section number",
      "title": "title of section",
      "why_it_applies": "1-2 sentences explaining how statutory elements match the user's stated facts.",
      "punishment": "quoted or summarized punishment from statutory text",
      "depends_on_missing_fact": "Specific factual condition needed, or null"
    }}
  ],
  "related_sections": [
    {{
      "act": "BNS",
      "section": "section number",
      "title": "title of section",
      "why_related": "Brief explanation of how it relates (e.g. dealing with stolen property afterwards)"
    }}
  ],
  "ambiguous_groups": [
    {{
      "label": "If you meant the old IPC section (IPC Section X)",
      "mapped_bns_section": "BNS Section Y",
      "title": "Title of BNS section",
      "details": "Explanation and punishment"
    }},
    {{
      "label": "If you meant Bharatiya Nyaya Sanhita (BNS Section X)",
      "mapped_bns_section": "BNS Section X",
      "title": "Title of BNS section",
      "details": "Explanation and punishment"
    }}
  ],
  "missing_facts_and_how_section_changes": [
    {{
      "missing_fact": "specific factual detail",
      "how_it_changes_section": "how that fact would change or escalate the section"
    }}
  ],
  "what_you_can_do": [
    {{
      "step": 1,
      "action": "Plain English practical step for victim (only for retrieved BNSS sections).",
      "section_basis": "BNSS Section 173(1)"
    }}
  ],
  "caveats": [
    "General caveat",
    "{DISCLAIMER_TEXT}"
  ]
}}"""

    user_prompt = f"""User Input: "{user_message}"

Analysis:
- Mode: {analysis.get('mode', 'situation')}
- Is Direct Lookup: {analysis.get('is_direct_lookup', False)}
- Candidate Offences: {', '.join(analysis.get('candidate_offences', []))}
- Ambiguity Notes: {analysis.get('ambiguity_notes', 'None')}

RETRIEVED STATUTORY CHUNKS:
{retrieved_text}

Generate the grounded JSON response strictly following all rules:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        
        # Ensure disclaimer is in caveats
        if "caveats" not in data or not isinstance(data["caveats"], list):
            data["caveats"] = []
        if not any("not constitute formal legal advice" in c for c in data["caveats"]):
            data["caveats"].append(DISCLAIMER_TEXT)
            
        data["is_fallback"] = False
        data["fallback_reason"] = ""
        return data
    except json.JSONDecodeError as jde:
        reason = f"JSON parse error from LLM response: {jde}"
        print(f"[FALLBACK TRIGGERED] {reason}")
        return _heuristic_answer(user_message, analysis, retrieved_chunks, fallback_reason=reason)
    except Exception as e:
        reason = f"OpenRouter API error: {type(e).__name__}: {str(e)}"
        print(f"[FALLBACK TRIGGERED] {reason}")
        return _heuristic_answer(user_message, analysis, retrieved_chunks, fallback_reason=reason)

def _heuristic_answer(user_message: str, analysis: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]], fallback_reason: str = "") -> Dict[str, Any]:
    """
    Structured fallback generator when offline or testing without LLM API key.
    Includes Civil-vs-Criminal check, ambiguous group separation, and element filtering.
    """
    msg_lower = user_message.lower()
    is_direct_lookup = analysis.get("is_direct_lookup", False) or analysis.get("mode") == "question"
    
    # ── 1. Check for Ambiguous Bare Section Numbers (e.g. "What is Section 302?") ──
    detected_secs = analysis.get("section_detection", {}).get("detected", [])
    ambiguous_item = next((d for d in detected_secs if d.get("status") == "ambiguous" and d.get("as_ipc") and d.get("as_bns")), None)
    
    ambiguous_groups = []
    if ambiguous_item and is_direct_lookup:
        num = ambiguous_item["number"]
        # Group 1: Old IPC Section
        ipc_matches = ambiguous_item.get("as_ipc", [])
        if ipc_matches:
            ipc_info = ipc_matches[0]
            bns_sec = ipc_info.get("bns_section", "")
            subj = ipc_info.get("subject", "")
            ambiguous_groups.append({
                "label": f"If you meant the old IPC section (IPC Section {num})",
                "mapped_bns_section": f"BNS Section {bns_sec}",
                "title": subj,
                "details": f"Under the old Indian Penal Code, Section {num} covered {subj}. In the new Bharatiya Nyaya Sanhita, 2023, this offence is now codified under BNS Section {bns_sec}."
            })
            
        # Group 2: New BNS Section
        bns_matches = ambiguous_item.get("as_bns", [])
        if bns_matches:
            bns_info = bns_matches[0]
            subj = bns_info.get("subject", "")
            ambiguous_groups.append({
                "label": f"If you meant Bharatiya Nyaya Sanhita (BNS Section {num})",
                "mapped_bns_section": f"BNS Section {num}",
                "title": subj,
                "details": f"Under the new Bharatiya Nyaya Sanhita, 2023, Section {num} addresses {subj}. (Note: This is completely distinct from old IPC Section {num})."
            })

    # ── 2. Civil vs. Criminal Check ──
    is_deposit_or_contract = any(w in msg_lower for w in ["deposit", "landlord", "security deposit", "rent", "tenant", "flat", "lease", "agreement", "contract", "refund"])
    has_clear_crime = any(w in msg_lower for w in ["stole", "snatch", "knife", "robbery", "beat", "weapon", "kill", "stab", "forged"])
    
    civil_check = {
        "is_likely_civil": False,
        "assessment": None,
        "usual_civil_remedy": None,
        "criminal_applicability": None
    }
    
    if is_deposit_or_contract and not has_clear_crime:
        civil_check["is_likely_civil"] = True
        civil_check["assessment"] = "The facts describe a tenancy or contractual dispute concerning the refund of a security deposit. Under settled Indian law, a dispute arising from breach of a lease agreement or monetary repayment is primarily a civil matter, unless fraudulent intention existed from the inception."
        civil_check["usual_civil_remedy"] = "Issue a formal Legal Demand Notice through an advocate calling for refund within 15 days. If unfulfilled, file a Summary Suit for Recovery of Money under Order XXXVII of the Code of Civil Procedure, 1908 (CPC), or initiate proceedings before the jurisdictional Rent Authority or District Consumer Commission for deficiency in service."
        civil_check["criminal_applicability"] = "Criminal sections such as BNS Section 316 (Criminal Breach of Trust) or BNS Section 318 (Cheating) are purely CONDITIONAL. They apply only if there is evidence of fraudulent inducement at the time the deposit was taken or dishonest conversion of the entrusted amount to the landlord's own use."

    # ── 3. Filter and Rank Candidate Offences ──
    applicable = []
    related_sections = []
    if not ambiguous_groups:
        offences_list = [c for c in retrieved_chunks if c.get("type") == "offence"]
        
        # Element matching filters:
        # Rule A: BNS Section 305 requires dwelling, building, means of transport (inside bus/vehicle/train), place of worship, or government property.
        # If incident happened at a "bus stop" or open street, BNS 305 elements DO NOT match!
        is_bus_stop_or_street = any(w in msg_lower for w in ["bus stop", "pavement", "street", "road", "footpath", "walking", "metro station platform"])
        is_inside_vehicle = any(w in msg_lower for w in ["inside the bus", "in a bus", "in a train", "in a car", "inside a cab", "in a taxi"])
        
        # Aftermath property sections (dealing with property afterwards):
        # e.g., BNS 252 (gift to recover stolen property), BNS 317 (stolen property), BNS 320 (receiving), BNS 323 (concealing)
        aftermath_sec_nums = {"252", "317", "320", "321", "322", "323"}
        
        primary_offences = []
        for c in offences_list:
            sec_num = str(c.get("section", "")).strip()
            # If theft at bus stop/street and not inside vehicle, reject BNS 305
            if sec_num == "305" and is_bus_stop_or_street and not is_inside_vehicle:
                continue
            if sec_num in aftermath_sec_nums or "stolen property" in c.get("title", "").lower() or "recover" in c.get("title", "").lower():
                related_sections.append({
                    "act": c.get("act", "BNS"),
                    "section": sec_num,
                    "title": c.get("title", "").strip(),
                    "why_related": "Deals with possessing, receiving, or recovering stolen property after the primary offence."
                })
            else:
                primary_offences.append(c)
                
        # If no primary offence was separated, keep them
        if not primary_offences and offences_list:
            primary_offences = offences_list
            
        def offence_relevance(c):
            score = 0
            title_lower = c.get("title", "").lower()
            sec = str(c.get("section", ""))
            for cand in analysis.get("candidate_offences", []):
                if cand.lower() in title_lower:
                    score += 15
            for q in analysis.get("search_queries", []):
                if sec in q:
                    score += 6
            score += c.get("rerank_score", 0.0) * 5
            return score
            
        primary_offences.sort(key=offence_relevance, reverse=True)
        
        for rank, c in enumerate(primary_offences[:3], 1):
            sec = c.get("section", "")
            act = c.get("act", "")
            title = c.get("title", "").strip()
            text = c.get("text", "")
            
            punishment = "Refer to statutory text."
            p_match = re.search(r"shall be punished with\s+([^\.]+)", text, re.IGNORECASE)
            if p_match:
                punishment = f"Shall be punished with {p_match.group(1).strip()}."
                
            # Grounded minimal explanation clearly labeled offline
            cand_name = analysis.get("candidate_offences", ["the alleged act"])[0]
            if is_direct_lookup:
                why_text = f"[Offline Mode] Section {sec} ({title}) of {act}, 2023 defines this statutory provision and punishment."
                depends_on = None
            elif civil_check["is_likely_civil"]:
                why_text = f"[Offline Mode] {act} Section {sec} ({title}) may conditionally apply only if proof of criminal fraudulent inducement or dishonest conversion is established."
                depends_on = "Requires proof of criminal dishonest intention from the inception (otherwise strictly a civil dispute)."
            else:
                why_text = f"[Offline Mode] Primary candidate offence matching '{cand_name}' based on statutory keywords. Review full text for exact ingredients."
                depends_on = None
                if rank > 1 and analysis.get("missing_facts"):
                    depends_on = analysis.get("missing_facts")[min(rank - 2, len(analysis.get("missing_facts")) - 1)]
                
            applicable.append({
                "rank": rank,
                "act": act,
                "section": sec,
                "title": title,
                "why_it_applies": why_text,
                "punishment": punishment,
                "depends_on_missing_fact": depends_on
            })

    # ── 4. Procedural Steps ("What you can do") ──
    # ONLY use procedural steps whose section was actually retrieved in retrieved_chunks!
    retrieved_bnss_sections = {
        str(c.get("section", "")).strip().lower(): c
        for c in retrieved_chunks
        if c.get("act") == "BNSS"
    }
    
    procedure_steps = []
    if not is_direct_lookup and not ambiguous_groups:
        if civil_check["is_likely_civil"]:
            # Civil dispute steps
            procedure_steps.append({
                "step": 1,
                "action": "Issue a formal Legal Demand Notice through an advocate demanding refund of the security deposit with interest within 15 days.",
                "section_basis": "Legal Notice / Indian Contract Act"
            })
            procedure_steps.append({
                "step": 2,
                "action": "If unrefunded, file a Summary Suit for Recovery of Money under Order XXXVII of the Code of Civil Procedure (CPC) or approach the Rent Authority.",
                "section_basis": "Order XXXVII, Code of Civil Procedure, 1908"
            })
            # Only include BNSS step if BNSS 173 was retrieved
            if any("173" in s for s in retrieved_bnss_sections):
                procedure_steps.append({
                    "step": 3,
                    "action": "Only if dishonest misappropriation or fraud from inception can be demonstrated, lodge a police complaint under BNSS Section 173(1) for Criminal Breach of Trust.",
                    "section_basis": "BNSS Section 173(1) (Conditional)"
                })
        else:
            step_num = 1
            # Step 1: FIR / Zero FIR (BNSS 173)
            if any("173" in s for s in retrieved_bnss_sections):
                procedure_steps.append({
                    "step": step_num,
                    "action": "Visit the nearest police station immediately to lodge a First Information Report (FIR). If outside the territorial jurisdiction, demand registration of a Zero FIR.",
                    "section_basis": "BNSS Section 173(1)"
                })
                step_num += 1
            # Step 2: Refusal complaint to SP (BNSS 173(4))
            if any("173(4)" in s or ("173" in s and "173(4)" in retrieved_bnss_sections[s].get("text", "")) for s in retrieved_bnss_sections):
                procedure_steps.append({
                    "step": step_num,
                    "action": "If the station officer refuses to record your FIR, send the substance of your complaint in writing by registered post or in person to the Superintendent of Police (SP).",
                    "section_basis": "BNSS Section 173(4)"
                })
                step_num += 1
            # Step 3: Magistrate application (BNSS 175(3) or 223)
            if any("175" in s or "223" in s for s in retrieved_bnss_sections):
                procedure_steps.append({
                    "step": step_num,
                    "action": "If no action is taken, file an application supported by affidavit before the Judicial Magistrate seeking an investigation order or file a private complaint.",
                    "section_basis": "BNSS Section 175(3) / Section 223"
                })
                step_num += 1

    missing_facts = []
    for mf in analysis.get("missing_facts", []):
        missing_facts.append({
            "missing_fact": mf,
            "how_it_changes_section": "Depending on this factual element, the offence may change, become aggravated, or determine whether criminal jurisdiction applies over civil remedies."
        })
        
    return {
        "civil_vs_criminal_check": civil_check,
        "applicable_sections": applicable,
        "related_sections": related_sections,
        "ambiguous_groups": ambiguous_groups,
        "missing_facts_and_how_section_changes": missing_facts,
        "what_you_can_do": procedure_steps,
        "caveats": [
            f"[OFFLINE HEURISTIC MODE] {fallback_reason or 'API key not configured'}. Results generated via offline rule-based heuristics.",
            DISCLAIMER_TEXT
        ],
        "is_fallback": True,
        "fallback_reason": fallback_reason
    }
        
    return {
        "civil_vs_criminal_check": civil_check,
        "applicable_sections": applicable,
        "ambiguous_groups": ambiguous_groups,
        "missing_facts_and_how_section_changes": missing_facts,
        "what_you_can_do": procedure_steps,
        "caveats": [
            "Grounded tentative legal assessment based strictly on retrieved BNS/BNSS statutory text.",
            DISCLAIMER_TEXT
        ],
        "is_fallback": True,
        "fallback_reason": fallback_reason
    }
