import re, json
from typing import List, Dict, Any
import torch

CLAUSE_LABELS = [
    "Definitions","Parties","Scope of Work","Fees and Payment","Taxes",
    "Term and Termination","Confidentiality","Data Protection/Privacy",
    "Intellectual Property","Warranties","Limitation of Liability",
    "Indemnity","Governing Law","Jurisdiction/Dispute Resolution",
    "Assignment/Subcontracting","Non-Solicitation","Service Levels",
    "Force Majeure","Audit/Inspection","Non-Compete","Publicity",
    "Notices","Entire Agreement","Miscellaneous","Other"
]

DOC_TYPES = [
    "NDA","SaaS MSA","Services Agreement","Employment Offer",
    "Consulting Agreement","Lease","Loan Agreement","Purchase Order",
    "Partnership Agreement","Privacy Policy","Terms of Service","Other"
]

def _apply_chat(model, tokenizer, user, system=None, temperature=0.5, max_new_tokens=512):
    """Chat wrapper for Granite"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.95,
            max_new_tokens=int(max_new_tokens)
        )

    gen = output[0, inputs.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def _ask_json(model, tokenizer, task, schema_desc, content, sys_prefix="", temp=0.45):
    sys_msg = f"""{sys_prefix}
You are performing the task: {task}.
Reply ONLY with a valid JSON object that matches this schema (no markdown, no prose):
{schema_desc}
"""
    out = _apply_chat(model, tokenizer, content, system=sys_msg, temperature=temp)
    txt = out.strip()

    if txt.startswith("```"):
        txt = txt.strip("` \n")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

    for _ in range(2):
        try:
            return json.loads(txt)
        except Exception:
            if "{" in txt and "}" in txt:
                txt = txt[txt.find("{"): txt.rfind("}")+1]
            else:
                break
    return {"error": "json_parse_error", "raw": out}

def extract_text_from_files(files) -> str:
    """Extracts text from uploaded PDFs, DOCX, TXT"""
    if not files:
        return ""
    chunks = []
    for f in files:
        name = (getattr(f, "name", None) or "").lower()
        try:
            if name.endswith(".pdf"):
                import pdfplumber
                with pdfplumber.open(f) as pdf:
                    chunks.append("\n".join(p.extract_text() or "" for p in pdf.pages))
            elif name.endswith(".docx"):
                from docx import Document
                d = Document(f)
                chunks.append("\n".join(p.text for p in d.paragraphs))
            else:
                chunks.append(f.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            chunks.append(f"[Error extracting {name}: {e}]")

    return "\n\n".join(chunks)[:200_000]

def split_clauses(text: str) -> List[str]:
    if not text.strip():
        return []
    text = text.replace("\r", "")
    text = re.sub(r"\n(?=\d+\.\s+[A-Z])", "\n§ ", text)
    text = re.sub(r"\n(?=[A-Z][A-Z \-/]{3,})", "\n§ ", text)
    text = re.sub(r"\n(?=Clause\s+\d+)", "\n§ ", text, flags=re.I)

    parts = [p.strip() for p in re.split(r"\n§ |\n{2,}", text) if p.strip()]
    clauses = []

    for p in parts:
        if len(p) > 1400:
            subs = re.split(r"(?<=[\.;:])\s+", p)
            buf = ""
            for s in subs:
                if len(buf) + len(s) < 1000:
                    buf += (" " if buf else "") + s
                else:
                    clauses.append(buf.strip())
                    buf = s
            if buf.strip():
                clauses.append(buf.strip())
        else:
            clauses.append(p)

    return [re.sub(r"\s+", " ", c).strip() for c in clauses]

def classify_document(model, tokenizer, full_text, hint="(auto)"):
    schema = '{"doc_type":"string","doc_summary":"string"}'
    prompt = (
        "Identify the legal document type and give a short summary.\n"
        f"Choose doc_type from: {DOC_TYPES}\n\n"
        f"Text:\n{full_text[:4000]}"
    )
    data = _ask_json(model, tokenizer, "Document Type", schema, prompt,
                     sys_prefix="You are a legal contract classifier.", temp=0.3)
    if "doc_type" not in data:
        return {"doc_type": "Other", "doc_summary": "(classification failed)"}
    return data

def analyze_clause(model, tokenizer, clause):
    schema = """
{
  "label": "string",
  "risk_score": "number",
  "summary": "string",
  "obligations": ["string"],
  "flags": ["string"]
}
"""
    prompt = (
        f"Analyze this contract clause:\n\n{clause}\n\n"
        "Return the JSON object."
    )
    return _ask_json(
        model, tokenizer, "Clause analysis", schema, prompt,
        sys_prefix="You are an expert contract lawyer.", temp=0.45
    )

def process_document(model, tokenizer, files):
    full_text = extract_text_from_files(files)
    meta = classify_document(model, tokenizer, full_text)
    clauses = split_clauses(full_text)

    results = []
    for idx, c in enumerate(clauses):
        r = analyze_clause(model, tokenizer, c)
        r["clause"] = c
        r["index"] = idx + 1
        results.append(r)

    return meta, results
