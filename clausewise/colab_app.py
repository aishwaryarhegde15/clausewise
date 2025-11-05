import os, io, re, json
from typing import List, Dict, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pypdf import PdfReader
import docx
import spacy
import gradio as gr

# Optional: sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_available = True
except Exception:
    _vader_available = False


# =============================
# Model config
# =============================
MODEL_ID = os.getenv("MODEL_ID", "ibm-granite/granite-3.2-2b-instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
if DEVICE != "cuda":
    model.to(DEVICE)
model.eval()

nlp = spacy.load("en_core_web_sm")


# =============================
# Helpers
# =============================
def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = (f"[SYSTEM]\n{system_prompt}\n" if system_prompt else "") + f"[USER]\n{user_prompt}\n[ASSISTANT]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if "[ASSISTANT]" in text:
        text = text.split("[ASSISTANT]")[-1].strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text.strip()


# =============================
# Document IO
# =============================
def load_text_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()

def load_text_from_docx(file_obj) -> str:
    data = file_obj.read()
    file_obj.seek(0)
    f = io.BytesIO(data)
    d = docx.Document(f)
    return "\n".join([p.text for p in d.paragraphs]).strip()

def load_text_from_txt(file_obj) -> str:
    data = file_obj.read()
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8", errors="ignore")
        except Exception:
            data = data.decode("latin-1", errors="ignore")
    return str(data).strip()

def load_document(file) -> str:
    if not file:
        return ""
    name = (file.name or "").lower()
    if name.endswith(".pdf"):
        return load_text_from_pdf(file)
    if name.endswith(".docx"):
        return load_text_from_docx(file)
    if name.endswith(".txt"):
        return load_text_from_txt(file)
    for fn in (load_text_from_pdf, load_text_from_docx, load_text_from_txt):
        try:
            return fn(file)
        except Exception:
            pass
    return ""

def get_text_from_inputs(file, text: str) -> str:
    file_text = load_document(file) if file else ""
    text = (text or "").strip()
    return file_text if len(file_text) > len(text) else text


# =============================
# Clause extraction
# =============================
CLAUSE_SPLIT_REGEX = re.compile(r"(?:(?:^\s*\d+(?:\.\d+)*[.)]\s+)|(?:^\s*[A-Z]\s*[.)]\s+)|(?:;?\s*\n))", re.MULTILINE)

def split_into_clauses(text: str, min_len: int = 40) -> List[str]:
    if not text:
        return []
    parts = re.split(CLAUSE_SPLIT_REGEX, text)
    if len(parts) < 2:
        parts = re.split(r"(?<=[.;])\s+\n?\s*", text)
    clauses = [p.strip() for p in parts if len(p.strip()) >= min_len]
    seen, unique = set(), []
    for c in clauses:
        key = re.sub(r"\s+", " ", c.lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# =============================
# Features
# =============================
def simplify_clause(clause: str) -> str:
    system = "You are a legal assistant that rewrites clauses into plain, layman-friendly English while preserving meaning. Flag key risks succinctly."
    user = f"Rewrite in plain English and list risks:\n\n{clause}"
    return llm_generate(system, user, max_new_tokens=400)

def ner_entities(text: str) -> Dict[str, List[str]]:
    if not text:
        return {}
    doc = nlp(text)
    out: Dict[str, List[str]] = {}
    for ent in doc.ents:
        out.setdefault(ent.label_, []).append(ent.text)
    return {k: sorted(set(v)) for k, v in out.items()}

def extract_clauses(text: str) -> List[str]:
    return split_into_clauses(text)

DOC_TYPES = [
    "Non-Disclosure Agreement (NDA)",
    "Lease Agreement",
    "Employment Contract",
    "Service Agreement",
    "Sales Agreement",
    "Consulting Agreement",
    "End User License Agreement (EULA)",
    "Terms of Service",
]

def classify_document_type(text: str) -> str:
    system = "You classify legal documents into a single best type from the list."
    labels = "\n".join(f"- {t}" for t in DOC_TYPES)
    user = f"Choose the best type from:\n{labels}\n\nDocument:\n{text[:5000]}"
    resp = llm_generate(system, user, max_new_tokens=200)
    for t in DOC_TYPES:
        if t.lower() in resp.lower():
            return t
    lower = text.lower()
    if any(k in lower for k in ["non-disclosure", "confidential", "nda"]):
        return "Non-Disclosure Agreement (NDA)"
    if any(k in lower for k in ["lease", "tenant", "landlord"]):
        return "Lease Agreement"
    if any(k in lower for k in ["employment", "employee"]):
        return "Employment Contract"
    if "services" in lower:
        return "Service Agreement"
    return "Service Agreement"

def negotiation_coach(clause: str) -> Tuple[str, List[Dict[str, Any]]]:
    system = "You are an AI negotiation coach."
    user = (
        "Propose 3 ranked alternative clauses with expected acceptance rates.\n"
        "Return JSON: {alternatives: [ {rank, acceptance_rate_percent, title, clause_text, rationale} ]}\n\n"
        f"Clause:\n{clause}"
    )
    resp = llm_generate(system, user, max_new_tokens=700)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        data = {"alternatives": []}
        for i in range(1, 4):
            data["alternatives"].append({
                "rank": i,
                "acceptance_rate_percent": max(50, 95 - (i - 1) * 10),
                "title": f"Alternative {i}",
                "clause_text": f"Alternative {i} based on the clause.",
                "rationale": "Heuristic fallback."
            })
    return json.dumps(data, indent=2), data["alternatives"]

def future_risk_predictor(clause: str) -> Tuple[str, List[Dict[str, Any]]]:
    system = "You forecast clause risks over 1-5 years."
    user = (
        "Return JSON: {timeline: [ {year: int, risk_score_0_100: int, key_risks: [str], mitigation: [str]} ]}.\n"
        f"Clause:\n{clause}"
    )
    resp = llm_generate(system, user, max_new_tokens=700)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        data = {"timeline": []}
        for y in range(1, 6):
            data["timeline"].append({
                "year": y,
                "risk_score_0_100": min(95, 40 + y * 8),
                "key_risks": ["Fallback risk"],
                "mitigation": ["Seek legal review", "Tighten definitions"],
            })
    return json.dumps(data, indent=2), data["timeline"]

def fairness_balance_meter(clause: str) -> Tuple[str, int, str]:
    system = "Score which party the clause favors: 0=A, 50=balanced, 100=B."
    user = "Return JSON: {score_0_100: int, rationale: str}.\n" + clause
    resp = llm_generate(system, user, max_new_tokens=300)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
        score = int(data.get("score_0_100", 50))
        rationale = data.get("rationale", "")
    except Exception:
        score, rationale = 50, "Fallback balanced score."
        data = {"score_0_100": score, "rationale": rationale}
    return json.dumps(data, indent=2), score, rationale

def clause_battle_arena(text_a: str, text_b: str) -> Tuple[str, str]:
    system = "Compare two contracts across key categories and pick a winner."
    user = (
        "Return JSON: {rounds: [ {category, winner: 'A'|'B'|'Draw', rationale} ], overall_winner: 'A'|'B'|'Draw', summary}.\n"
        f"Document A:\n{text_a[:4000]}\n\nDocument B:\n{text_b[:4000]}"
    )
    resp = llm_generate(system, user, max_new_tokens=900)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        data = {
            "rounds": [{"category": c, "winner": "Draw", "rationale": "Fallback"} for c in ["Liability", "Termination", "IP", "Payment", "Confidentiality", "Governing Law"]],
            "overall_winner": "Draw",
            "summary": "Fallback",
        }
    pretty = json.dumps(data, indent=2)
    rounds_md = "\n".join([f"- {r['category']}: {r['winner']} — {r.get('rationale','')}" for r in data.get("rounds", [])])
    md = f"Overall Winner: {data.get('overall_winner','Draw')}\n\nRounds:\n{rounds_md}\n\nSummary:\n{data.get('summary','')}"
    return pretty, md

PII_REGEXES = {
    "Email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "Phone": r"\+?\d[\d\-\s]{7,}\d",
    "SSN (US)": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card": r"\b(?:\d[ -]*?){13,16}\b",
}

def sensitive_data_sniffer(text: str) -> Tuple[str, Dict[str, List[str]]]:
    system = "Find privacy traps and personal data in the text."
    user = "Return JSON: {data_categories: [str], sharing_parties: [str], processing_purposes: [str], risks: [str], recommendations: [str]}.\n" + text[:6000]
    resp = llm_generate(system, user, max_new_tokens=700)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        data = {
            "data_categories": ["Name", "Email"],
            "sharing_parties": ["Service Provider"],
            "processing_purposes": ["Service delivery"],
            "risks": ["Over-collection"],
            "recommendations": ["Narrow purpose", "Limit retention"],
        }
    regex_hits: Dict[str, List[str]] = {}
    for label, pattern in PII_REGEXES.items():
        hits = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if hits:
            regex_hits[label] = sorted(set([h.strip() for h in hits]))
    return json.dumps({"llm": data, "regex_hits": regex_hits}, indent=2), regex_hits

def litigation_risk_radar(text: str) -> Tuple[str, str]:
    clauses = split_into_clauses(text)
    sample = "\n\n".join(clauses[:8]) if clauses else text[:4000]
    system = "Identify clauses likely to trigger disputes and give sample scenarios."
    user = "Return JSON: {hotspots: [ {clause_excerpt, risk_level: 'Low'|'Medium'|'High', why, sample_dispute_scenario} ]}.\n" + sample
    resp = llm_generate(system, user, max_new_tokens=900)
    try:
        data = json.loads(re.search(r"\{[\s\S]*\}", resp).group(0))
    except Exception:
        data = {
            "hotspots": [
                {
                    "clause_excerpt": (clauses[0][:280] if clauses else text[:280]),
                    "risk_level": "Medium",
                    "why": "Ambiguity",
                    "sample_dispute_scenario": "Missed milestone dispute",
                }
            ]
        }
    pretty = json.dumps(data, indent=2)
    md = "\n".join([
        f"- [{h.get('risk_level','Medium')}] {h.get('clause_excerpt','')}\n  Why: {h.get('why','')}\n  Scenario: {h.get('sample_dispute_scenario','')}"
        for h in data.get("hotspots", [])
    ])
    return pretty, md


# =============================
# Sentiment + chatbot
# =============================
if _vader_available:
    _sent = SentimentIntensityAnalyzer()
else:
    _sent = None

def analyze_sentiment(text: str) -> Dict[str, Any]:
    if not _sent:
        return {"label": "neutral", "compound": 0.0, "intensity": "mild", "scores": {}}
    scores = _sent.polarity_scores(text or "")
    compound = float(scores.get("compound", 0.0))
    if compound >= 0.3:
        label = "positive"
    elif compound <= -0.3:
        label = "negative"
    else:
        label = "neutral"
    intensity = "strong" if abs(compound) >= 0.6 else ("moderate" if abs(compound) >= 0.3 else "mild")
    return {"label": label, "compound": compound, "intensity": intensity, "scores": scores}

def legal_chatbot_reply(history: List[Dict[str, str]], user_text: str, context_text: str = "", temperature: float = 0.3) -> Tuple[str, Dict[str, Any]]:
    sent = analyze_sentiment(user_text)
    compound_val = float(sent.get("compound", 0.0))
    sent_label = sent.get("label", "neutral")
    sent_intensity = sent.get("intensity", "mild")
    tone_instr = {
        "negative": "Open with one empathetic sentence. Be calm, concise, action‑oriented. Offer 3 next steps.",
        "neutral":  "Be clear and structured with numbered guidance.",
        "positive": "Be encouraging; confirm understanding; give concise best‑practice tips.",
    }.get(sent_label, "Be clear and structured.")

    system = (
        "You are a careful legal research assistant. Use plain English and cite provided text when relevant. "
        "You are not a lawyer; this is not legal advice. "
        f"User sentiment: {sent_label} ({sent_intensity}, compound={compound_val:.2f}). "
        f"Adjust tone: {tone_instr}"
    )
    msgs = [{"role": "system", "content": system}]
    if context_text:
        msgs.append({"role": "system", "content": f"Relevant context:\n{context_text[:8000]}"})
    for m in history or []:
        if m.get("role") in ("user", "assistant"):
            msgs.append(m)
    msgs.append({"role": "user", "content": user_text})

    try:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in msgs]) + "\n[ASSISTANT]\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if "[ASSISTANT]" in text:
        text = text.split("[ASSISTANT]")[-1].strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    disclaimer = "\n\nNote: I’m an AI, not a lawyer. This is educational information, not legal advice."
    if disclaimer not in text:
        text += disclaimer
    return text, {"label": sent_label, "compound": compound_val, "intensity": sent_intensity}


# =============================
# UI
# =============================
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
CSS = """
#topbar { position: sticky; top: 0; z-index: 100; background: var(--body-background-fill); padding: 6px 0 8px; border-bottom: 1px solid #1f2937; }
.container { max-width: 1200px !important; margin: 0 auto !important; }
.card { background: #0f172a; border: 1px solid #1f2937; padding: 16px; border-radius: 12px; }
.small { color: #9ca3af; font-size: 12px; }
"""

with gr.Blocks(title="ClauseWise – Granite 3.2 (2B)", theme=theme, css=CSS) as demo:
    with gr.Column(elem_id="topbar", elem_classes=["container"]):
        gr.Markdown("### ClauseWise — Contract Intelligence (IBM Granite 3.2 2B)")
        gr.Markdown("Upload once at the top; then use any tool tab below. New: Legal Chatbot (sentiment‑aware).")

    with gr.Column(elem_classes=["container", "card"]):
        with gr.Row():
            doc_input = gr.File(label="Upload PDF / DOCX / TXT (optional)", file_count="single", file_types=[".pdf", ".docx", ".txt"])
            text_input = gr.Textbox(label="Or paste text", lines=8, placeholder="Paste clause or document")
        with gr.Row():
            clear_inputs = gr.Button("Clear Inputs", variant="secondary")
            gr.Markdown("<span class='small'>If both upload and paste are provided, the app uses whichever contains more text.</span>")

    with gr.Column(elem_classes=["container"]):
        with gr.Tabs():
            with gr.Tab("Legal Chatbot (Beta)"):
                chatbot = gr.Chatbot(height=420, type="messages", label="Ask legal questions (not legal advice)")
                chat_query = gr.Textbox(placeholder="Ask in natural language…", lines=2, label="Your Question")
                with gr.Row():
                    chat_temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Creativity")
                    chat_clear = gr.Button("Clear Chat", variant="secondary")
                    chat_send = gr.Button("Send", variant="primary")
                sentiment_out = gr.Label(label="Detected Sentiment")
                chat_state = gr.State([])

            with gr.Tab("Clause Simplification"):
                clause_in = gr.Textbox(label="Clause (optional, uses top input if empty)", lines=6)
                btn_simplify = gr.Button("Simplify Clause", variant="primary")
                clause_out = gr.Textbox(label="Plain English", lines=12)

            with gr.Tab("Named Entity Recognition"):
                btn_ner = gr.Button("Run NER", variant="primary")
                ner_out = gr.JSON(label="Entities (JSON)")

            with gr.Tab("Clause Extraction & Breakdown"):
                btn_extract = gr.Button("Extract Clauses", variant="primary")
                clauses_out = gr.Dataframe(headers=["Clause"], datatype=["str"], row_count=8, col_count=1, wrap=True)

            with gr.Tab("Document Type Classification"):
                btn_classify = gr.Button("Classify Document", variant="primary")
                class_out = gr.Textbox(label="Predicted Type", lines=2)

            with gr.Tab("AI Negotiation Coach"):
                coach_clause = gr.Textbox(label="Clause to Optimize (optional)", lines=6)
                btn_coach = gr.Button("Suggest Alternatives", variant="primary")
                coach_json = gr.JSON(label="Alternatives (JSON)")
                coach_table = gr.Dataframe(headers=["Rank", "Acceptance %", "Title", "Clause Text", "Rationale"], row_count=3, col_count=5, wrap=True)

            with gr.Tab("Future Risk Predictor"):
                risk_clause = gr.Textbox(label="Clause (optional)", lines=6)
                btn_risk = gr.Button("Predict 1–5 Year Risks", variant="primary")
                risk_json = gr.JSON(label="Timeline (JSON)")
                risk_table = gr.Dataframe(headers=["Year", "Risk Score (0–100)", "Key Risks", "Mitigation"], row_count=5, col_count=4, wrap=True)

            with gr.Tab("Fairness Balance Meter"):
                fairness_clause = gr.Textbox(label="Clause (optional)", lines=6)
                btn_fair = gr.Button("Compute Fairness", variant="primary")
                fairness_json = gr.JSON(label="Result (JSON)")
                fairness_score = gr.Slider(label="Balance Score (0=Party A, 50=Balanced, 100=Party B)", minimum=0, maximum=100, value=50, interactive=False)
                fairness_notes = gr.Textbox(label="Rationale / Notes", lines=4)

            with gr.Tab("Clause Battle Arena"):
                left_doc = gr.Textbox(label="Document A", lines=10)
                right_doc = gr.Textbox(label="Document B", lines=10)
                btn_battle = gr.Button("Compare", variant="primary")
                battle_json = gr.JSON(label="Battle (JSON)")
                battle_md = gr.Markdown()

            with gr.Tab("Sensitive Data Sniffer"):
                btn_sniff = gr.Button("Detect Privacy Traps & PII", variant="primary")
                sniff_json = gr.JSON(label="LLM Findings + Regex Hits (JSON)")
                pii_table = gr.Dataframe(headers=["PII Type", "Examples"], datatype=["str", "str"], row_count=6, col_count=2, wrap=True)

            with gr.Tab("Litigation Risk Radar"):
                btn_lit = gr.Button("Analyze Hotspots", variant="primary")
                lit_json = gr.JSON(label="Hotspots (JSON)")
                lit_md = gr.Markdown()

        gr.Markdown("<span class='small'>Disclaimer: Educational tool only. Not legal advice.</span>")

    def _ctx(file, text):
        return get_text_from_inputs(file, text)

    def chat_respond(chat_hist, query, file, text, temperature):
        if not (query and str(query).strip()):
            return (chat_hist or []), (chat_hist or []), gr.update(value=""), None
        history = chat_hist or []
        ctx = _ctx(file, text)
        reply, sent = legal_chatbot_reply(history, str(query).strip(), ctx, temperature=float(temperature))
        new_history = history + [
            {"role": "user", "content": str(query).strip()},
            {"role": "assistant", "content": reply},
        ]
        sent_label = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}.get(sent.get("label", "neutral"), "Neutral")
        compound_val = float(sent.get("compound", 0.0))
        sent_payload = {"Detected": sent_label, "compound": round(compound_val, 3)}
        return new_history, new_history, gr.update(value=""), sent_payload

    def chat_clear_fn():
        return [], [], gr.update(value=None)

    # Chat wiring
    chat_send.click(chat_respond, [chat_state, chat_query, doc_input, text_input, chat_temperature], [chatbot, chat_state, chat_query, sentiment_out])
    chat_query.submit(chat_respond, [chat_state, chat_query, doc_input, text_input, chat_temperature], [chatbot, chat_state, chat_query, sentiment_out])
    chat_clear.click(chat_clear_fn, outputs=[chatbot, chat_state, sentiment_out])

    # Tools wiring
    btn_simplify.click(lambda f, t, c: simplify_clause((c or _ctx(f, t)).strip()), [doc_input, text_input, clause_in], [clause_out])
    btn_ner.click(lambda f, t: ner_entities(_ctx(f, t)[:12000]), [doc_input, text_input], [ner_out])
    btn_extract.click(lambda f, t: [[c] for c in extract_clauses(_ctx(f, t))], [doc_input, text_input], [clauses_out])
    btn_classify.click(lambda f, t: classify_document_type(_ctx(f, t)), [doc_input, text_input], [class_out])

    def _coach(f, t, c):
        base = (c or _ctx(f, t)).strip()
        pretty, alts = negotiation_coach(base)
        table = [[a.get("rank", ""), a.get("acceptance_rate_percent", ""), a.get("title", ""), a.get("clause_text", ""), a.get("rationale", "")] for a in alts]
        return pretty, table
    btn_coach.click(_coach, [doc_input, text_input, coach_clause], [coach_json, coach_table])

    def _risk(f, t, c):
        base = (c or _ctx(f, t)).strip()
        pretty, timeline = future_risk_predictor(base)
        rows = [[x.get("year", ""), x.get("risk_score_0_100", ""), "; ".join(x.get("key_risks", [])), "; ".join(x.get("mitigation", []))] for x in timeline]
        return pretty, rows
    btn_risk.click(_risk, [doc_input, text_input, risk_clause], [risk_json, risk_table])

    def _fair(f, t, c):
        base = (c or _ctx(f, t)).strip()
        pretty, score, rationale = fairness_balance_meter(base)
        return pretty, score, rationale
    btn_fair.click(_fair, [doc_input, text_input, fairness_clause], [fairness_json, fairness_score, fairness_notes])

    btn_battle.click(lambda a, b: clause_battle_arena(a, b), [left_doc, right_doc], [battle_json, battle_md])
    btn_sniff.click(lambda f, t: (lambda p, r: (p, [[k, "; ".join(v)] for k, v in r.items()]))(*sensitive_data_sniffer(_ctx(f, t))), [doc_input, text_input], [sniff_json, pii_table])
    btn_lit.click(lambda f, t: litigation_risk_radar(_ctx(f, t)), [doc_input, text_input], [lit_json, lit_md])

    clear_inputs.click(lambda: (None, ""), outputs=[doc_input, text_input])

demo.queue(max_size=64).launch(share=True, inline=False, show_error=True)


