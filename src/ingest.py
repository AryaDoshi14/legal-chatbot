"""
PDF text extraction and section-level chunker for BNS and BNSS.
Extracts sections with full metadata: act, section, title, chapter, type, source_file.
Keeps illustrations, explanations, exceptions, sub-sections together in the section chunk.
"""

import os
import re
import json
import pymupdf
from typing import List, Dict, Any, Optional

def clean_text_whitespace(text: str) -> str:
    """Normalize whitespace and strip spurious lines/nulls."""
    lines = [l.strip() for l in text.split("\n")]
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    return "\n".join(cleaned_lines).strip()

def clean_section_text(text: str) -> str:
    """
    Strip footnote numbers, standalone page numbers, footnote attribution lines,
    and trailing next-chapter headers from section text.
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        l_str = line.strip()
        if not l_str:
            cleaned_lines.append("")
            continue
            
        # Standalone page numbers (e.g. "177", "178")
        if re.match(r"^\d{1,4}$", l_str):
            continue
            
        # Footnote attribution lines (e.g. "399 Section 378, IPC, 1860.", "1. Subs. by...")
        if re.match(r"^\d{1,4}\s+(?:Section\s+\d+|Subs\.|Ins\.|Omitted|The|See)\b", l_str):
            continue
        if re.match(r"^\d{1,4}\s+IPC,\s*\d{4}\.?$", l_str):
            continue
            
        # Strip trailing footnote numbers attached after punctuation (e.g. "theft. 399", "fine: 400")
        l_cleaned = re.sub(r"(?<=[\.:,;])\s*\d{2,4}\b", "", l_str)
        l_cleaned = re.sub(r"(?<=[a-zA-Z])\s*\d{3,4}\b", "", l_cleaned)
        
        cleaned_lines.append(l_cleaned)
        
    # Strip any trailing Chapter headers belonging to the next chapter
    while cleaned_lines and (
        not cleaned_lines[-1].strip()
        or re.match(r"^CHAPTER\s+[IVXLCDM]+", cleaned_lines[-1].strip(), re.IGNORECASE)
        or re.match(r"^OF\s+[A-Z\s\(\)]+$", cleaned_lines[-1].strip())
    ):
        cleaned_lines.pop()
        
    return clean_text_whitespace("\n".join(cleaned_lines))

def is_header_or_footer(b: tuple) -> bool:
    """Check if block is in header or footer margin."""
    y0, y1 = b[1], b[3]
    txt = b[4].strip()
    if y0 < 52:
        return True
    if y1 > 770:
        return True
    if txt.isdigit() and (y0 < 65 or y1 > 740):
        return True
    if "THE BHARATIYA NYAYA SANHITA" in txt and y0 < 65:
        return True
    if "THE GAZETTE OF INDIA" in txt and y0 < 65:
        return True
    return False

def parse_bns(pdf_path: str = "data/bns.pdf") -> List[Dict[str, Any]]:
    """
    Parse BNS handbook PDF into section chunks.
    BNS Bare act begins at page 24 (index 23).
    """
    doc = pymupdf.open(pdf_path)
    sections = []
    current_chapter = "PRELIMINARY"
    
    raw_pages_text = []
    for p_idx in range(23, len(doc)):
        page = doc[p_idx]
        blocks = page.get_text("blocks")
        page_lines = []
        for b in blocks:
            if is_header_or_footer(b):
                continue
            txt = b[4].strip()
            if not txt:
                continue
            ch_match = re.search(r"CHAPTER\s+([IVXLCDM]+)[\s\n]*(?:\([^)]+\))?[\s\n]*([A-Z\s,\'\"()–—\-]+)?", txt)
            if ch_match:
                ch_num = ch_match.group(1)
                ch_title = ch_match.group(2).strip() if ch_match.group(2) else ""
                current_chapter = f"CHAPTER {ch_num} - {ch_title}".strip(" -")
            
            for line in txt.split("\n"):
                l_str = line.strip()
                if l_str:
                    page_lines.append((l_str, p_idx + 1, current_chapter))
        raw_pages_text.append(page_lines)

    all_lines = [item for sublist in raw_pages_text for item in sublist]
    sec_markers = []
    
    for idx, (line, page_num, chapter) in enumerate(all_lines):
        line_clean = line.strip()
        
        # Special Case: Section 61 (Criminal Conspiracy)
        if re.match(r"^61\.\s*\(", line_clean):
            sec_markers.append((idx, "61", "CRIMINAL CONSPIRACY", page_num, chapter))
            continue
            
        # Special Case: Section 349 (Selling goods marked with a counterfeit property mark)
        if re.match(r"^349\.[A-Z]", line_clean):
            title = re.sub(r"^349\.", "", line_clean).strip()
            sec_markers.append((idx, "349", title, page_num, chapter))
            continue
            
        # Standard BNS section header:
        # e.g. "318. CHEATING." or "SECTION 3. GENERAL EXPLANATIONS."
        m = re.match(r"^(?:SECTION\s+)?(\d+[A-Z]?)\.\s*([A-Z0-9\s,\'\"()–—\-/\\]*)", line_clean)
        if m:
            sec_num = m.group(1)
            raw_title = m.group(2).strip() if m.group(2) else ""
            clean_title = re.sub(r"\d+$", "", raw_title).strip()
            
            # If title is on next line (e.g. line was "181." or "348.")
            if len(clean_title) < 2 and idx + 1 < len(all_lines):
                next_l = all_lines[idx + 1][0].strip()
                if next_l.isupper() and not next_l.startswith("(") and len(next_l) > 2:
                    clean_title = next_l
                    
            alpha = [c for c in clean_title if c.isalpha()]
            # Ensure title is overwhelmingly uppercase (filters out lowercase IPC comparison table rows)
            if len(alpha) >= 3 and (sum(1 for c in alpha if c.isupper()) / len(alpha)) > 0.8:
                if sec_num.isdigit() and 1 <= int(sec_num) <= 358:
                    if "Section" not in line_clean and "IPC" not in line_clean and not line_clean.endswith("IPC, 1860."):
                        sec_markers.append((idx, sec_num, clean_title, page_num, chapter))

    # Keep unique section numbers
    seen_secs = set()
    unique_markers = []
    for m in sec_markers:
        idx, sec_num, clean_title, page_num, chapter = m
        if sec_num not in seen_secs:
            seen_secs.add(sec_num)
            unique_markers.append(m)

    unique_markers.sort(key=lambda x: x[0])
    
    for i in range(len(unique_markers)):
        start_idx, sec_num, title, page_num, chapter = unique_markers[i]
        end_idx = unique_markers[i + 1][0] if i + 1 < len(unique_markers) else len(all_lines)
        
        chunk_lines = [all_lines[k][0] for k in range(start_idx, end_idx)]
        full_text = "\n".join(chunk_lines).strip()
        
        title = re.sub(r"[\.0-9]+$", "", title).strip()
        if not title:
            title = f"Section {sec_num}"

        sec_chunk = {
            "act": "BNS",
            "section": sec_num,
            "title": title.title(),
            "chapter": chapter,
            "type": "offence",
            "source_file": "data/bns.pdf",
            "page_start": page_num,
            "text": clean_text_whitespace(full_text)
        }
        sections.append(sec_chunk)

    return sections

def parse_bnss(pdf_path: str = "data/bnss.pdf") -> List[Dict[str, Any]]:
    """
    Parse BNSS Gazette PDF into section chunks.
    Extracts margin titles and aligns with section bodies.
    """
    doc = pymupdf.open(pdf_path)
    sections = []
    current_chapter = "PRELIMINARY"
    
    all_pages = []
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        blocks = page.get_text("blocks")
        
        margin_notes = []
        main_lines = []
        
        for b in blocks:
            if is_header_or_footer(b):
                continue
            txt = b[4].strip()
            if not txt:
                continue
            
            if b[0] < 110 or b[0] > 480:
                margin_notes.append({
                    "y0": b[1],
                    "y1": b[3],
                    "text": " ".join(txt.split())
                })
            else:
                ch_m = re.match(r"^CHAPTER\s+([IVXLCDM]+)[\s\n]+([A-Z\s,\'\"()–—\-]+)", txt)
                if ch_m:
                    current_chapter = f"CHAPTER {ch_m.group(1)} - {ch_m.group(2).strip()}"
                
                for line in txt.split("\n"):
                    l_str = line.strip()
                    if l_str:
                        main_lines.append((l_str, b[1], p_idx + 1, current_chapter))
                        
        all_pages.append({
            "page_num": p_idx + 1,
            "margin_notes": margin_notes,
            "main_lines": main_lines
        })
        
    all_lines = []
    for p in all_pages:
        for item in p["main_lines"]:
            all_lines.append(item)
            
    sec_markers = []
    for idx, (line, y0, page_num, chapter) in enumerate(all_lines):
        m = re.match(r"^(\d+[A-Z]?)\.\s+(.*)", line)
        if m:
            sec_num = m.group(1)
            if sec_num.isdigit() and 1 <= int(sec_num) <= 535:
                page_data = all_pages[page_num - 1]
                best_title = ""
                min_dist = 9999
                for mn in page_data["margin_notes"]:
                    dist = abs(mn["y0"] - y0)
                    if dist < 65 and dist < min_dist:
                        min_dist = dist
                        best_title = mn["text"]
                        
                sec_markers.append((idx, sec_num, best_title, page_num, chapter))

    seen_secs = set()
    unique_markers = []
    for m in sec_markers:
        idx, sec_num, title, page_num, chapter = m
        if sec_num not in seen_secs:
            seen_secs.add(sec_num)
            unique_markers.append(m)

    unique_markers.sort(key=lambda x: x[0])
    
    for i in range(len(unique_markers)):
        start_idx, sec_num, title, page_num, chapter = unique_markers[i]
        end_idx = unique_markers[i + 1][0] if i + 1 < len(unique_markers) else len(all_lines)
        
        chunk_lines = [all_lines[k][0] for k in range(start_idx, end_idx)]
        full_text = "\n".join(chunk_lines).strip()
        
        if not title:
            title = f"Section {sec_num}"
            
        title = re.sub(r"[\.0-9]+$", "", title).strip()

        sec_chunk = {
            "act": "BNSS",
            "section": sec_num,
            "title": title,
            "chapter": chapter,
            "type": "procedure",
            "source_file": "data/bnss.pdf",
            "page_start": page_num,
            "text": clean_text_whitespace(full_text)
        }
        sections.append(sec_chunk)

    return sections

def split_long_chunks(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> List[Dict[str, Any]]:
    """
    Split only very long sections, keeping act/section/title in every piece.
    """
    final_chunks = []
    for c in chunks:
        text = c["text"]
        if len(text) <= max_chars:
            chunk_copy = dict(c)
            chunk_copy["chunk_id"] = f"{c['act']}_{c['section']}"
            final_chunks.append(chunk_copy)
        else:
            paras = text.split("\n\n")
            current_piece = []
            curr_len = 0
            part = 1
            header_prefix = f"[{c['act']} Section {c['section']}: {c['title']}]\n"
            
            for p in paras:
                if curr_len + len(p) > max_chars and current_piece:
                    piece_text = header_prefix + "\n\n".join(current_piece)
                    chunk_copy = dict(c)
                    chunk_copy["text"] = piece_text
                    chunk_copy["part"] = part
                    chunk_copy["chunk_id"] = f"{c['act']}_{c['section']}_p{part}"
                    final_chunks.append(chunk_copy)
                    part += 1
                    current_piece = [p]
                    curr_len = len(p)
                else:
                    current_piece.append(p)
                    curr_len += len(p)
                    
            if current_piece:
                piece_text = (header_prefix if part > 1 else "") + "\n\n".join(current_piece)
                chunk_copy = dict(c)
                chunk_copy["text"] = piece_text
                chunk_copy["part"] = part
                chunk_copy["chunk_id"] = f"{c['act']}_{c['section']}_p{part}" if part > 1 else f"{c['act']}_{c['section']}"
                final_chunks.append(chunk_copy)
                
    return final_chunks

def load_and_parse_all(save_json: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """Parse both BNS and BNSS, returning chunks categorized into offences and procedures."""
    print("Parsing BNS...")
    bns_sections = parse_bns("data/bns.pdf")
    print(f"Extracted {len(bns_sections)} BNS sections.")
    
    print("Parsing BNSS...")
    bnss_sections = parse_bnss("data/bnss.pdf")
    print(f"Extracted {len(bnss_sections)} BNSS sections.")
    
    bns_chunks = split_long_chunks(bns_sections)
    bnss_chunks = split_long_chunks(bnss_sections)
    
    if save_json:
        os.makedirs("data", exist_ok=True)
        with open("data/bns_chunks.json", "w", encoding="utf-8") as f:
            json.dump(bns_chunks, f, indent=2, ensure_ascii=False)
        with open("data/bnss_chunks.json", "w", encoding="utf-8") as f:
            json.dump(bnss_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(bns_chunks)} BNS chunks to data/bns_chunks.json")
        print(f"Saved {len(bnss_chunks)} BNSS chunks to data/bnss_chunks.json")
        
    return {
        "offences": bns_chunks,
        "procedures": bnss_chunks
    }

if __name__ == "__main__":
    load_and_parse_all()
