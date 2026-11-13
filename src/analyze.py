"""
Query analyzer with IPC/BNS ambiguity rule and OpenRouter LLM call.
Extracts structured facts (what, where, how, who), candidate offences,
and plain legal wording search queries (NO section numbers unless user wrote them).
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAPPING_FILE = "data/ipc_bns_mapping.json"
_MAPPING_CACHE = None

def load_mapping():
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None and os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            _MAPPING_CACHE = json.load(f)
    return _MAPPING_CACHE

def detect_sections(user_text: str) -> Dict[str, Any]:
    """
    Detect section numbers the user mentions.
    Ambiguity rule: A bare number (e.g. "302") is ambiguous only if it exists in BOTH acts.
    Numbers > 358 exist exclusively in IPC (BNS only has 358 sections).
    """
    mapping = load_mapping()
    ipc_to_bns = mapping.get("ipc_to_bns", {}) if mapping else {}
    bns_to_ipc = mapping.get("bns_to_ipc", {}) if mapping else {}
    
    text_lower = user_text.lower()
    has_ipc_keyword = bool(re.search(r"\b(ipc|indian penal code)\b", text_lower))
    has_bns_keyword = bool(re.search(r"\b(bns|bharatiya nyaya sanhita|nyaya sanhita)\b", text_lower))
    
    # Match numbers that could be sections (1 to 511)
    potential_nums = re.findall(r"\b\d+[A-Za-z]?\b", user_text)
    detected = []
    
    for num in potential_nums:
        if num in ["2023", "2024", "1860", "2025", "2026", "1", "2", "3", "4", "5"] and not re.search(rf"\b(?:section|sec|s\.)\s*{num}\b", text_lower):
            continue
            
        ipc_matches = ipc_to_bns.get(num, [])
        bns_matches = bns_to_ipc.get(num, [])
        
        if not ipc_matches and not bns_matches:
            continue
            
        pattern_ipc_nearby = re.search(rf"(?:ipc|indian penal code)[\s,\.:-]*{num}|{num}[\s,\.:-]*(?:ipc|indian penal code)", text_lower)
        pattern_bns_nearby = re.search(rf"(?:bns|nyaya sanhita)[\s,\.:-]*{num}|{num}[\s,\.:-]*(?:bns|nyaya sanhita)", text_lower)
        
        if pattern_ipc_nearby or (has_ipc_keyword and not has_bns_keyword):
            status = "ipc"
        elif pattern_bns_nearby or (has_bns_keyword and not has_ipc_keyword):
            status = "bns"
        elif ipc_matches and not bns_matches:
            status = "ipc"
        elif bns_matches and not ipc_matches:
            status = "bns"
        else:
            status = "ambiguous"
            
        info = {
            "number": num,
            "status": status,
            "as_ipc": ipc_matches,
            "as_bns": bns_matches
        }
        detected.append(info)
        
    return {
        "has_ipc_keyword": has_ipc_keyword,
        "has_bns_keyword": has_bns_keyword,
        "detected": detected
    }

def analyze_query(user_message: str) -> Dict[str, Any]:
    """
    Analyze user input message using Claude/OpenRouter and output structured JSON.
    Uses plain legal wording in search queries without section numbers unless user cited them.
    """
    section_detection = detect_sections(user_message)
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _heuristic_analysis(user_message, section_detection)
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    context_notes = []
    for d in section_detection["detected"]:
        num = d["number"]
        if d["status"] == "ipc":
            bns_info = [f"BNS {x['bns_raw']} ({x['subject']})" for x in d['as_ipc']]
            context_notes.append(f"User cited Section {num} as IPC. Maps to: {'; '.join(bns_info)}.")
        elif d["status"] == "bns":
            ipc_info = [f"IPC {x['ipc_raw']} ({x['subject']})" for x in d['as_bns']]
            context_notes.append(f"User cited Section {num} as BNS. Corresponds to old: {'; '.join(ipc_info)}.")
        else:
            bns_info = [f"BNS {x['bns_raw']} ({x['subject']})" for x in d['as_ipc']]
            ipc_info = [f"Old IPC {x['ipc_raw']} ({x['subject']})" for x in d['as_bns']]
            context_notes.append(
                f"Section {num} is AMBIGUOUS. If IPC {num} -> {'; '.join(bns_info)}. "
                f"If BNS {num} -> {'; '.join(ipc_info)}."
            )
            
    context_str = "\n".join(context_notes) if context_notes else "No specific section numbers explicitly mentioned."
    
    system_prompt = """You are an expert Indian Criminal Law assistant specializing in the Bharatiya Nyaya Sanhita, 2023 (BNS) and Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS).
Analyze the user's message and return a strictly valid JSON object.

The output JSON must contain EXACTLY these keys:
- "mode": "question" or "situation"
- "is_direct_lookup": boolean (true if asking what a section means or punishment for a section)
- "facts": a concise list of extracted factual elements (e.g. ["What: purse", "Where: bus stop", "How: stolen without consent", "Who: unknown person"]). Do NOT copy the user's full sentence. Empty list if direct question.
- "candidate_offences": list of string candidate offences under BNS (e.g. ["Theft", "Snatching", "Extortion", "Cheating"])
- "search_queries": list of 3 to 5 targeted, high-precision legal search queries using plain statutory wording only (e.g. "dishonestly takes movable property out of possession without consent", "information in cognizable cases first information report registration"). DO NOT include section numbers in search queries unless the user explicitly wrote a section number in their message.
- "missing_facts": list of string factual elements that would change which section applies (e.g. presence of force, weapon, daytime/nighttime, value, relationship, consent)
- "ambiguity_notes": string summary of any section ambiguity detected or empty string

Rules:
1. Use plain legal statutory terminology only for search queries. NO section numbers in search queries unless user explicitly provided them.
2. If user mentions an old IPC section, translate it to BNS candidate offences.
3. Output JSON ONLY, no conversational filler or markdown code fences."""

    user_prompt = f"""User message: "{user_message}"

Detected Section Numbers & Ambiguity Context:
{context_str}

Analyze this message now and return ONLY the JSON object:"""

    model_name = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
    print(f"[LLM Analyze] Model: {model_name} | Temperature: 0.0")
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=1000,
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
        data["section_detection"] = section_detection
        return data
    except Exception as e:
        print(f"LLM analyze error ({e}), falling back to heuristic.")
        res = _heuristic_analysis(user_message, section_detection)
        res["llm_error"] = str(e)
        return res

def _heuristic_analysis(user_message: str, section_detection: Dict[str, Any]) -> Dict[str, Any]:
    """Robust fallback when offline or testing without LLM API."""
    msg = user_message.lower()
    is_question = bool(re.search(r"^(what|is|how|can|who|punishment|bailable|cognizable|\?|define)\b", msg) or "?" in msg)
    mode = "question" if is_question else "situation"
    
    is_direct_lookup = bool(
        re.search(r"\b(what is section|punishment for section|define section|what does section|meaning of section)\b", msg)
        or (is_question and any(d["number"] for d in section_detection.get("detected", [])))
    )
    
    candidates = []
    queries = []
    missing_facts = []
    
    # Extract structured facts (what, where, how, who)
    facts = []
    if mode == "situation":
        what = None
        for item in ["purse", "handbag", "wallet", "phone", "mobile", "chain", "gold", "jewelry", "car", "bike", "deposit", "security deposit", "money", "cash", "advance", "property", "wheat", "bag"]:
            if item in msg:
                what = item
                break
        if what:
            facts.append(f"What: {what}")
            
        where = None
        for loc in ["bus stop", "metro", "flat", "apartment", "house", "shop", "farmhouse", "street", "road", "public way", "restaurant", "hospital", "bank", "online", "telegram"]:
            if loc in msg:
                where = loc
                break
        if where:
            facts.append(f"Where: {where}")
            
        how = None
        if "knifepoint" in msg or "knife" in msg:
            how = "taken at knifepoint under threat of violence"
        elif "grabbed" in msg or "snatch" in msg:
            how = "grabbed/snatched suddenly from person"
        elif "stolen" in msg or "steal" in msg:
            how = "stolen dishonestly without consent"
        elif "refused to return" in msg or "refused to refund" in msg:
            how = "withheld/refused to refund after vacating"
        elif "scam" in msg or "fraud" in msg or "cheat" in msg or "duped" in msg:
            how = "deceived and induced into paying money"
        elif "forged" in msg:
            how = "signature forged on documents"
        elif "hit" in msg or "beat" in msg:
            how = "inflicted physical hurt/blows"
        elif "followed" in msg or "stalk" in msg:
            how = "followed repeatedly without consent"
        if how:
            facts.append(f"How: {how}")
            
        who = None
        for person in ["two men on a motorcycle", "two guys on a bike", "three armed men", "friend", "landlord", "tenant", "business partner", "colleague", "neighbor", "customer", "driver", "warehouse manager", "employee"]:
            if person in msg:
                who = person
                break
        if not who:
            who = "unknown person/accused"
        facts.append(f"Who: {who}")
    
    # Plain statutory search queries without section numbers unless user cited them
    is_robbery = any(w in msg for w in ["robbery", "armed", "gunpoint", "knifepoint", "knife", "weapon", "dacoity", "looted", "cashier", "forced me to give"])
    if is_robbery:
        candidates.append("Robbery")
        queries.append("robbery theft or extortion causing fear of instant hurt or death")
        queries.append("attempt to commit robbery armed with deadly weapon")
        queries.append("dishonestly takes movable property out of possession without consent")
        queries.append("information in cognizable cases first information report registration")
        missing_facts.append("Whether deadly weapon was used or death/grievous hurt was attempted")

    is_snatching = any(w in msg for w in ["snatch", "snatched", "snatching", "chain", "grab", "grabbed", "grabbing"])
    if is_snatching and not is_robbery:
        candidates.append("Snatching")
        queries.append("snatching sudden or quickly seizure of property from person")
        queries.append("dishonestly takes movable property out of possession without consent")
        queries.append("theft with hurt or wrongful restraint robbery")
        queries.append("information in cognizable cases first information report registration")
        missing_facts.append("Whether any hurt or bodily pain was caused during the snatching")

    if any(w in msg for w in ["stolen", "steal", "theft", "purse", "wallet", "bag", "phone", "bike", "car", "metro"]):
        if not is_robbery and not is_snatching:
            candidates.append("Theft")
            queries.append("dishonestly takes movable property out of possession without consent")
            queries.append("theft of movable property without consent of owner")
            queries.append("information in cognizable cases first information report registration")
            missing_facts.append("Whether the property was taken with or without force/threat (snatching vs robbery vs theft)")
            missing_facts.append("Whether the theft occurred in a building/house/conveyance or open place")
            missing_facts.append("Value of stolen property (if under 5000 rupees and first offence, community service may apply under BNS 303(2))")
        elif is_snatching:
            candidates.append("Theft")
        elif is_robbery:
            candidates.append("Extortion")

    if any(w in msg for w in ["left behind", "left their bag", "left a bag", "left his bag", "left her bag", "forgot their bag", "finder", "found property", "found a"]):
        candidates.append("Dishonest Misappropriation of Property")
        queries.append("dishonest misappropriation of property conversion to own use finder")
        queries.append("dishonestly takes movable property out of possession without consent")
        queries.append("criminal breach of trust entrusted with property")
        missing_facts.append("Whether the finder knew or had means of discovering the owner before converting the property")

    if any(w in msg for w in ["cheat", "cheating", "fraud", "scam", "deceive", "duped", "otp", "advance", "blocked me", "420", "job scam", "online job", "part-time job", "work from home"]):
        candidates.append("Cheating")
        queries.append("cheating and dishonestly inducing delivery of property")
        queries.append("cheating by personation pretending to be another")
        queries.append("complaint to magistrate inquiry into offence")
        missing_facts.append("Whether fraudulent inducement existed from inception")

    if any(w in msg for w in ["entrust", "entrusted", "gave my car", "deposit", "landlord", "security deposit", "sold it", "scrap", "warehouse", "misappropriat", "406"]):
        candidates.append("Criminal Breach of Trust")
        queries.append("criminal breach of trust entrusted with property dishonest conversion")
        queries.append("dishonest misappropriation of property conversion to own use")
        queries.append("cheating dishonestly inducing delivery of property")
        queries.append("information in cognizable cases first information report registration")
        missing_facts.append("Whether property/deposit was handed over under a contractual tenancy/agreement and whether dishonest conversion occurred")

    if any(w in msg for w in ["extort", "extortion", "protection money", "hafta", "ransom"]):
        candidates.append("Extortion")
        queries.append("extortion putting person in fear of injury to deliver property")
        queries.append("information in cognizable cases first information report registration")
        missing_facts.append("Whether the fear was of death, grievous hurt, or property damage")

    # Criminal Intimidation / Threats (Check before actual homicide if only a threat)
    is_threat = any(w in msg for w in ["threat", "threatened", "threatening", "intimidat", "threatens"])
    if is_threat:
        candidates.append("Criminal Intimidation")
        queries.append("criminal intimidation threat of injury to person reputation or property")
        queries.append("criminal intimidation by threatening to cause death or grievous hurt")
        queries.append("information in cognizable cases first information report registration")
        missing_facts.append("Whether the threat was to cause death, grievous hurt, or destruction of property")

    if any(w in msg for w in ["kill", "murder", "death", "stab", "302"]):
        if not is_threat:
            candidates.append("Murder")
            queries.append("punishment for murder intentional causing of death culpable homicide")
            queries.append("culpable homicide not amounting to murder")
            queries.append("arrest procedure for murder cognizable offence")
            missing_facts.append("Intention and presence of grave and sudden provocation")

    if any(w in msg for w in ["hit", "fracture", "beat", "iron rod", "baseball bat", "slap", "pipe", "injury", "hurt"]):
        candidates.append("Hurt / Grievous Hurt")
        queries.append("voluntarily causing hurt or grievous hurt with dangerous weapon")
        queries.append("assault or use of criminal force")
        queries.append("information in cognizable cases first information report registration")

    if any(w in msg for w in ["defam", "defamation", "defamatory", "reputation", "linkedin", "facebook", "post"]):
        candidates.append("Defamation")
        queries.append("defamation words spoken or published intended to harm reputation")
        queries.append("complaint to magistrate inquiry into offence")

    if any(w in msg for w in ["stalk", "stalking", "follow", "following", "instagram", "spy camera", "trial room", "voyeurism"]):
        if any(w in msg for w in ["spy camera", "trial room", "changing room", "voyeurism"]):
            candidates.append("Voyeurism")
            queries.append("voyeurism capture image woman in private act")
        else:
            candidates.append("Stalking")
            queries.append("stalking follow woman electronic communication monitoring")
        queries.append("information in cognizable cases first information report registration")

    if any(w in msg for w in ["smashed", "windshield", "vandal", "damage", "scratched", "mischief"]):
        candidates.append("Mischief")
        queries.append("mischief causing destruction of property damage wrongful loss")
        queries.append("information in cognizable cases first information report registration")

    if any(w in msg for w in ["trespass", "unlawfully entered", "courtyard", "vacate", "locked me out"]):
        candidates.append("Criminal Trespass")
        queries.append("criminal trespass house-trespass unlawful entry into property")
        queries.append("wrongful restraint prevent proceeding in any direction")

    if any(w in msg for w in ["forged", "forgery", "fake signature", "cheque"]):
        candidates.append("Forgery")
        queries.append("forgery making false document valuable security")
        queries.append("cheating dishonestly inducing delivery of property")

    if any(w in msg for w in ["cyclist", "pedestrian", "hit and run", "ran over", "speeding"]):
        candidates.append("Rash driving and Causing death by negligence")
        queries.append("rash driving or riding on a public way")
        queries.append("causing death by negligence hit and run escape")

    # Section queries ONLY if user cited section numbers explicitly
    for d in section_detection["detected"]:
        num = d["number"]
        if d["status"] == "ipc":
            for m in d["as_ipc"]:
                bns_sec = m["bns_section"]
                queries.append(f"BNS section {bns_sec} {m['subject']}")
                candidates.append(m['subject'])
        elif d["status"] == "bns":
            queries.append(f"BNS section {num}")
        else: # ambiguous
            for m in d["as_ipc"]:
                queries.append(f"BNS section {m['bns_section']} {m['subject']}")
            for m in d["as_bns"]:
                queries.append(f"BNS section {num} {m['subject']}")

    # Deduplicate queries while preserving priority order
    seen_q = set()
    deduped_queries = []
    for q in queries:
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            deduped_queries.append(q)

    if not deduped_queries:
        deduped_queries = [
            f"Indian criminal law offence procedure {user_message}",
            "information in cognizable cases first information report registration",
            "complaint to magistrate inquiry into offence"
        ]
        
    return {
        "mode": mode,
        "is_direct_lookup": is_direct_lookup,
        "facts": facts,
        "candidate_offences": candidates or ["Criminal Offence"],
        "search_queries": deduped_queries[:5],
        "missing_facts": missing_facts,
        "ambiguity_notes": "Ambiguous section numbers detected" if any(d["status"] == "ambiguous" for d in section_detection["detected"]) else "",
        "section_detection": section_detection
    }
