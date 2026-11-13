"""
Legal RAG Chatbot – Indian Criminal Law (BNS + BNSS)
Streamlit UI: section cards, source text, procedural steps, dark-mode readable text, bottom disclaimer.
"""

import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyze import analyze_query
from src.retrieve import retrieve
from src.answer import generate_answer, DISCLAIMER_TEXT
from src.verify import verify_citations

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal RAG Chatbot – BNS & BNSS",
    page_icon="⚖️",
    layout="wide",
)

# ── Custom CSS: Strict high-contrast dark text on light cards for Dark Theme compatibility ──
st.markdown("""
<style>
/* High contrast styling readable in both light and dark Streamlit themes */
.disclaimer-box {
    background-color: #fef3c7 !important;
    border: 1px solid #f59e0b !important;
    border-radius: 8px;
    padding: 12px 18px;
    margin-top: 24px;
    margin-bottom: 20px;
    color: #1f2937 !important;
    font-size: 0.92em;
    line-height: 1.5;
}
.disclaimer-box strong {
    color: #92400e !important;
}

.section-card {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 5px solid #2563eb !important;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    color: #0f172a !important;
}
.section-card * {
    color: #0f172a !important;
}
.section-card h4 {
    margin: 0 0 8px 0;
    color: #1d4ed8 !important;
    font-weight: 700;
}
.section-card .punishment {
    background-color: #e0f2fe !important;
    border: 1px solid #bae6fd !important;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 0.92em;
    color: #0369a1 !important;
}
.section-card .punishment strong {
    color: #075985 !important;
}
.section-card .conditional-note {
    background-color: #fef2f2 !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 4px;
    padding: 6px 10px;
    margin-top: 8px;
    font-size: 0.88em;
    color: #991b1b !important;
}

.step-card {
    background-color: #f0fdf4 !important;
    border: 1px solid #dcfce7 !important;
    border-left: 5px solid #16a34a !important;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #0f172a !important;
    line-height: 1.5;
}
.step-card * {
    color: #0f172a !important;
}
.step-card strong {
    color: #15803d !important;
}

.missing-fact {
    background-color: #fffbeb !important;
    border: 1px solid #fef3c7 !important;
    border-left: 5px solid #f59e0b !important;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #0f172a !important;
    line-height: 1.45;
}
.missing-fact * {
    color: #0f172a !important;
}
.missing-fact strong {
    color: #b45309 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("⚖️ Legal RAG Chatbot")
st.caption("Indian Criminal Law — Bharatiya Nyaya Sanhita (BNS) & Bharatiya Nagarik Suraksha Sanhita (BNSS)")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper: render a structured answer ───────────────────────────────────────
def render_answer(result: dict):
    """Render verified answer as section cards, steps, missing facts, and verification info."""

    # 1. Fallback Warning if LLM did not complete normally
    if result.get("is_fallback"):
        reason = result.get("fallback_reason", "LLM API unavailable")
        st.warning(f"⚠️ **Offline Heuristic Mode Active:** {reason}. Responses are generated via structured rule-based statutory matching rather than live LLM synthesis.")

    civil_check = result.get("civil_vs_criminal_check", {})
    if civil_check and civil_check.get("is_likely_civil"):
        st.markdown(
            f'<div style="background-color: #eff6ff !important; border: 1px solid #bfdbfe !important; border-left: 5px solid #2563eb !important; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; color: #1e3a8a !important;">'
            f'<h4 style="margin: 0 0 6px 0; color: #1d4ed8 !important; font-weight: 700;">⚖️ Civil vs. Criminal Assessment: Primarily Civil Dispute</h4>'
            f'<p style="color: #1e3a8a !important; margin-bottom: 8px;">{civil_check.get("assessment")}</p>'
            f'<div style="background-color: #dbeafe !important; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px; color: #1e40af !important;">'
            f'<strong>📜 Usual Civil Remedy:</strong> {civil_check.get("usual_civil_remedy")}</div>'
            f'<div style="background-color: #fef3c7 !important; padding: 8px 12px; border-radius: 6px; color: #92400e !important;">'
            f'<strong>⚠️ Criminal Section Applicability:</strong> {civil_check.get("criminal_applicability")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    ambiguous_groups = result.get("ambiguous_groups", [])
    if ambiguous_groups:
        st.subheader("🔀 Ambiguous Section Number: Comparative Analysis")
        st.info("The section number you queried is ambiguous between the old IPC and the new BNS. Below are the two distinct legal statutory provisions:")
        for grp in ambiguous_groups:
            label = grp.get("label", "")
            mapped = grp.get("mapped_bns_section", "")
            title = grp.get("title", "")
            details = grp.get("details", "")
            src_text = grp.get("source_text", "")
            st.markdown(
                f'<div class="section-card">'
                f'<h4>{label}</h4>'
                f'<p style="font-weight: 600; color: #2563eb !important;">{mapped}: {title}</p>'
                f'<p>{details}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            if src_text:
                with st.expander(f"📄 Statutory Source Text — {mapped}"):
                    st.text(src_text[:3000])

    sections = result.get("applicable_sections", [])
    steps = result.get("what_you_can_do", [])
    missing = result.get("missing_facts_and_how_section_changes", [])
    caveats = result.get("caveats", [])
    verification = result.get("verification_summary", {})

    # ── Applicable sections ──────────────────────────────────────────────
    if sections and not ambiguous_groups:
        st.subheader("📋 Applicable Sections (Ranked Candidates)")
        show_rank = len(sections) > 1
        for sec in sections:
            rank = sec.get("rank")
            rank_str = f"Rank {rank}: " if (show_rank and rank) else ""
            act = sec.get("act", "")
            num = sec.get("section", "")
            title = sec.get("title", "")
            why = sec.get("why_it_applies", "")
            punishment = sec.get("punishment", "")
            depends = sec.get("depends_on_missing_fact")
            source_text = sec.get("source_text", "")
            verified = sec.get("verified", False)

            badge = "✅ Verified (Top-5 Grounded)" if verified else "⚠️ Unverified"
            
            cond_html = f'<div class="conditional-note"><strong>⚠️ Conditional Factor:</strong> {depends}</div>' if depends else ""

            st.markdown(
                f'<div class="section-card">'
                f'<h4>{rank_str}{act} Section {num} — {title} <small style="font-size: 0.75em; font-weight: normal; color: #475569 !important;">({badge})</small></h4>'
                f'<p>{why}</p>'
                f'<div class="punishment">🔨 <strong>Statutory Punishment:</strong> {punishment}</div>'
                f'{cond_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if source_text:
                with st.expander(f"📄 Full Statutory Text — {act} Section {num}"):
                    st.text(source_text[:3000])
    elif not ambiguous_groups:
        st.info("No applicable sections found in the loaded law text for this query.")

    # ── Related sections ─────────────────────────────────────────────────
    related = result.get("related_sections", [])
    if related:
        st.subheader("🔗 Related Sections (Property Aftermath / Recovery)")
        for r_sec in related:
            r_act = r_sec.get("act", "BNS")
            r_num = r_sec.get("section", "")
            r_title = r_sec.get("title", "")
            r_why = r_sec.get("why_related", "")
            st.markdown(
                f'<div class="section-card">'
                f'<h4>{r_act} Section {r_num} — {r_title}</h4>'
                f'<p>{r_why}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── What you can do ──────────────────────────────────────────────────
    if steps:
        st.subheader("🚶 What You Can Do (Procedural Steps)")
        for step in steps:
            step_num = step.get("step", "")
            action = step.get("action", "")
            basis = step.get("section_basis", "")
            step_verified = step.get("verified", False)
            step_source = step.get("source_text", "")
            v_badge = "✅" if step_verified else ""

            st.markdown(
                f'<div class="step-card">'
                f'<strong>Step {step_num}:</strong> {action} '
                f'<em>(Basis: {basis})</em> {v_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if step_source:
                with st.expander(f"📄 Procedure Source Text — {basis}"):
                    st.text(step_source[:2000])

    # ── Missing facts ────────────────────────────────────────────────────
    if missing:
        st.subheader("❓ Details That Would Change or Escalate the Charge")
        for mf in missing:
            fact = mf.get("missing_fact", "")
            change = mf.get("how_it_changes_section", "")
            st.markdown(
                f'<div class="missing-fact">'
                f'<strong>Fact: {fact}</strong><br/>{change}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Caveats ──────────────────────────────────────────────────────────
    clean_caveats = [c.strip() for c in caveats if c and c.strip() and "formal legal advice" not in c.lower()]
    if clean_caveats:
        with st.expander("📌 Legal Caveats & Statutory Notes"):
            for c in clean_caveats:
                st.markdown(f"- {c}")

    # ── Verification summary ─────────────────────────────────────────────
    if verification:
        total = verification.get("total_cited", 0)
        verified_count = verification.get("verified_count", 0)
        unverified = verification.get("unverified_citations", [])
        if unverified:
            st.warning(
                f"⚠️ Relevance Alert: {len(unverified)} cited section(s) were removed because they were not confirmed in the top 5 reranked statutory chunks: "
                + ", ".join(unverified)
            )
        elif total > 0:
            st.caption(f"🛡️ Deterministic Citation Verifier: All {verified_count}/{total} cited sections verified within the top 5 retrieved statutory chunks.")

    # ── Visible Bottom Disclaimer ────────────────────────────────────────
    st.markdown(
        f'<div class="disclaimer-box">⚠️ <strong>Statutory Disclaimer:</strong> {DISCLAIMER_TEXT}</div>',
        unsafe_allow_html=True,
    )

# ── Chat history display ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("result"), dict):
            render_answer(msg["result"])
        else:
            st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Describe your situation or ask a legal question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process through pipeline
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal elements…"):
            analysis = analyze_query(user_input)

        queries = analysis.get("search_queries", [user_input])

        with st.spinner("Searching BNS & BNSS databases…"):
            retrieved = retrieve(queries, user_situation=user_input, top_k_rerank=8)

        with st.spinner("Generating grounded legal assessment…"):
            answer = generate_answer(user_input, analysis, retrieved)

        with st.spinner("Verifying statutory citations (Top-5 check)…"):
            verified = verify_citations(answer, retrieved)

        render_answer(verified)

        # Store for chat history replay
        st.session_state.messages.append({
            "role": "assistant",
            "content": "",
            "result": verified,
        })
