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
# Model config - OPTIMIZED
# =============================
import time
print("🚀 Starting ClauseWise initialization...")
start_time = time.time()

MODEL_ID = os.getenv("MODEL_ID", "ibm-granite/granite-3.2-2b-instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"📦 Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True,
    cache_dir=None  # Use default cache
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded!")

print(f"🤖 Loading model on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_cache=True,  # Enable KV cache for faster generation
)
if DEVICE != "cuda":
    model.to(DEVICE)
model.eval()

# Optimize for inference
if hasattr(torch, "compile") and DEVICE == "cuda":
    try:
        print("⚡ Compiling model for faster inference...")
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compiled!")
    except Exception as e:
        print(f"⚠️ Compilation skipped: {e}")

print(f"✅ Model loaded in {time.time() - start_time:.2f}s")

print("📚 Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("✅ All models loaded! Ready to use.")
print(f"\n{'='*60}")
print("🎉 ClauseWise is ready!")
print(f"{'='*60}")
print(f"📍 Device: {DEVICE}")
print(f"⚡ Model: {MODEL_ID}")
print(f"💾 Load time: {time.time() - start_time:.2f}s")
print(f"\n💡 Tips for faster responses:")
print("   - Keep questions concise")
print("   - Use lower creativity (0.1-0.3) for faster responses")
print("   - Responses typically take 3-10 seconds")
print(f"{'='*60}\n")


# =============================
# Helpers
# =============================
def llm_generate(system_prompt: str, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.3, top_p: float = 0.9) -> str:
    """Optimized generation with faster settings"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = (f"[SYSTEM]\n{system_prompt}\n" if system_prompt else "") + f"[USER]\n{user_prompt}\n[ASSISTANT]\n"
    
    # Truncate prompt if too long
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    # Optimized generation parameters for speed
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=min(int(max_new_tokens), 384),  # Cap at 384 for speed
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=temperature > 0.1,  # Faster if low temperature
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
            num_beams=1 if temperature > 0.1 else 1,  # Greedy for speed
            early_stopping=True,
        )
    
    # Decode only new tokens (faster)
    new_tokens = out_ids[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up
    if "[ASSISTANT]" in text:
        text = text.split("[ASSISTANT]")[-1].strip()
    
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
    """Fixed chatbot that works like a conversation - understands context and gives natural responses."""
    sent = analyze_sentiment(user_text)
    compound_val = float(sent.get("compound", 0.0))
    sent_label = sent.get("label", "neutral")
    sent_intensity = sent.get("intensity", "mild")
    
    # Build conversation history properly
    msgs = []
    
    # System prompt - be conversational and helpful
    system_prompt = (
        "You are ClauseWise, a friendly and knowledgeable legal AI assistant powered by IBM Granite 3.2 2B. "
        "You help users understand legal documents, clauses, contracts, and answer legal questions in plain English. "
        "You are conversational, clear, and helpful. When users ask questions, you respond naturally like a helpful assistant would. "
        "You are NOT a lawyer and this is NOT legal advice - always remind users to consult a qualified attorney for important decisions. "
        "If the user has uploaded a document, use it to provide context-specific answers. "
        "Be friendly, professional, and use examples when helpful."
    )
    msgs.append({"role": "system", "content": system_prompt})
    
    # Add document context if available
    if context_text and len(context_text.strip()) > 0:
        msgs.append({
            "role": "system", 
            "content": f"User has uploaded a document. Here's the relevant content:\n\n{context_text[:8000]}\n\nUse this context to answer questions about the document."
        })
    
    # Add conversation history (only user and assistant messages, skip system)
    for msg in (history or []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    
    # Add current user message
    msgs.append({"role": "user", "content": user_text.strip()})

    # Generate response
    try:
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback prompt format
        prompt_parts = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

    # Optimize input length
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    # Faster generation for chatbot
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=min(384, 512),  # Cap for speed
            temperature=float(temperature),
            top_p=0.9,
            do_sample=temperature > 0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,  # Greedy decoding for speed
            early_stopping=True,
        )
    
    # Decode only new tokens (much faster)
    new_tokens = out_ids[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response
    text = text.split("\n\nUser:")[0].strip()  # Remove any trailing user text
    text = text.split("\nSystem:")[0].strip()   # Remove any trailing system text
    text = text.split("\nAssistant:")[0].strip()  # Remove any duplicate assistant markers
    
    # Add disclaimer if not present
    disclaimer = "\n\n⚠️ *Note: I'm an AI assistant, not a lawyer. This is educational information, not legal advice. Please consult a qualified attorney for important decisions.*"
    if "not a lawyer" not in text.lower() and "legal advice" not in text.lower():
        text += disclaimer
    
    return text, {"label": sent_label, "compound": compound_val, "intensity": sent_intensity}


# =============================
# UI - COLORFUL & INTERACTIVE
# =============================
theme = gr.themes.Monochrome().set(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md"
)

CSS = """
/* Vibrant colorful theme */
:root {
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --gradient-4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    --gradient-5: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

#topbar {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
    backdrop-filter: blur(10px);
    position: sticky;
    top: 0;
    z-index: 1000;
    padding: 20px 0;
    border-bottom: 3px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 20px;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255, 255, 255, 0.3);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.gradio-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

.gradio-button.secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4) !important;
}

.gradio-tabs {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 15px !important;
    padding: 10px !important;
    margin: 20px 0 !important;
}

.gradio-tabs .tabitem {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    border-radius: 10px !important;
    margin: 5px !important;
    padding: 10px 20px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.gradio-tabs .tabitem:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
}

.gradio-tabs .tabitem.selected {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
    box-shadow: 0 4px 15px rgba(67, 233, 123, 0.5) !important;
}

.gradio-textbox textarea {
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.gradio-chatbot {
    border-radius: 20px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
}

h1, h2, h3 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
}

.small {
    color: rgba(102, 126, 234, 0.8);
    font-size: 13px;
}

.gradio-slider input[type="range"] {
    accent-color: #667eea;
}

.gradio-file {
    border: 2px dashed rgba(102, 126, 234, 0.4) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

.gradio-file:hover {
    border-color: #667eea !important;
    background: rgba(102, 126, 234, 0.05) !important;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.loading {
    animation: pulse 2s ease-in-out infinite;
}
"""

with gr.Blocks(title="ClauseWise – Granite 3.2 (2B)", theme=theme, css=CSS) as demo:
    with gr.Column(elem_id="topbar", elem_classes=["container"]):
        gr.Markdown(
            """
            # 🎯 ClauseWise — AI Legal Assistant
            ### Powered by IBM Granite 3.2 2B Instruct
            **Your intelligent partner for contract analysis, clause simplification, and legal Q&A**
            """,
            elem_classes=["card"]
        )
        gr.Markdown(
            "💡 **Pro Tip:** Upload a document or paste text above, then ask any legal question in the chatbot below!",
            elem_classes=["card"]
        )

    with gr.Column(elem_classes=["container", "card"]):
        gr.Markdown("### 📄 Document Input")
        with gr.Row():
            doc_input = gr.File(
                label="📎 Upload PDF / DOCX / TXT",
                file_count="single",
                file_types=[".pdf", ".docx", ".txt"],
                elem_classes=["card"]
            )
            text_input = gr.Textbox(
                label="📝 Or paste text directly",
                lines=8,
                placeholder="Paste your contract, clause, or legal document here...",
                elem_classes=["card"]
            )
        with gr.Row():
            clear_inputs = gr.Button("🔄 Clear All Inputs", variant="secondary", size="lg")
            gr.Markdown(
                "<span class='small'>💡 If both upload and paste are provided, the app uses whichever contains more text.</span>"
            )

    with gr.Column(elem_classes=["container"]):
        with gr.Tabs():
            with gr.Tab("💬 Legal Chatbot (Ask Anything!)"):
                gr.Markdown(
                    """
                    ### 🤖 Interactive Legal Assistant
                    **Ask me anything about contracts, clauses, legal terms, risks, negotiations, or your uploaded document!**
                    
                    I'm powered by IBM Granite 3.2 2B and I understand context, remember our conversation, and give you helpful, plain-English answers.
                    """,
                    elem_classes=["card"]
                )
                gr.Markdown(
                    "⏱️ **Note:** Responses typically take 3-10 seconds. Please be patient while the AI generates your answer.",
                    elem_classes=["card"]
                )
                chatbot = gr.Chatbot(
                    height=500,
                    type="messages",
                    label="💬 Conversation",
                    show_copy_button=True,
                    elem_classes=["card"],
                    loading_message="🤔 Thinking... Please wait..."
                )
                chat_query = gr.Textbox(
                    placeholder="💭 Ask anything... e.g., 'What does this clause mean?', 'Is this contract fair?', 'What are the risks here?'",
                    lines=3,
                    label="Your Question",
                    elem_classes=["card"]
                )
                with gr.Row():
                    chat_temperature = gr.Slider(
                        0.0, 1.0,
                        value=0.3,
                        step=0.05,
                        label="🎨 Creativity Level",
                        info="Lower = more focused & faster, Higher = more creative"
                    )
                    chat_clear = gr.Button("🗑️ Clear Chat History", variant="secondary", size="lg")
                    chat_send = gr.Button("🚀 Send Message", variant="primary", size="lg")
                sentiment_out = gr.Label(
                    label="😊 Detected Sentiment",
                    elem_classes=["card"]
                )
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
        """Chatbot response handler with error handling"""
        if not (query and str(query).strip()):
            return (chat_hist or []), (chat_hist or []), gr.update(value=""), None
        
        try:
            history = chat_hist or []
            ctx = _ctx(file, text)
            
            # Generate response (this may take a few seconds)
            reply, sent = legal_chatbot_reply(history, str(query).strip(), ctx, temperature=float(temperature))
            
            # Build new history
            new_history = history + [
                {"role": "user", "content": str(query).strip()},
                {"role": "assistant", "content": reply},
            ]
            
            # Sentiment payload
            sent_label = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}.get(sent.get("label", "neutral"), "Neutral")
            compound_val = float(sent.get("compound", 0.0))
            sent_payload = {"Detected": sent_label, "compound": round(compound_val, 3)}
            
            return new_history, new_history, gr.update(value=""), sent_payload
            
        except Exception as e:
            # Error handling
            error_msg = f"⚠️ Error: {str(e)}\n\nPlease try again or rephrase your question."
            history = chat_hist or []
            new_history = history + [
                {"role": "user", "content": str(query).strip()},
                {"role": "assistant", "content": error_msg},
            ]
            return new_history, new_history, gr.update(value=""), {"Detected": "Error", "compound": 0.0}

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


