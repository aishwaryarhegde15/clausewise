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
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
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
        # Handle both file paths (strings) and file objects
        if isinstance(f, str):
            # f is a file path
            file_path = f
            name = file_path.lower()
        else:
            # f is a file object
            file_path = getattr(f, "name", None)
            name = (file_path or "").lower()
        
        if not file_path:
            continue
            
        try:
            if name.endswith(".pdf"):
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    chunks.append("\n".join(p.extract_text() or "" for p in pdf.pages))
            elif name.endswith(".docx"):
                from docx import Document
                d = Document(file_path)
                chunks.append("\n".join(p.text for p in d.paragraphs))
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
                    chunks.append(txt_file.read())
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

# Cache for storing previously analyzed clauses
_clause_cache = {}

def analyze_clause_batch(model, tokenizer, clauses):
    """Analyze multiple clauses in a single batch for better performance."""
    # Check cache first
    cached_results = []
    uncached_clauses = []
    uncached_indices = []
    
    for i, clause in enumerate(clauses):
        clause_hash = hash(clause.strip())
        if clause_hash in _clause_cache:
            cached_results.append((i, _clause_cache[clause_hash]))
        else:
            uncached_clauses.append((i, clause, clause_hash))
    
    if not uncached_clauses:
        # All results were cached, return in order
        return [res for _, res in sorted(cached_results, key=lambda x: x[0])]
    
    # Process uncached clauses in batches
    schema = """
[
  {
    "label": "string",
    "risk_score": "number",
    "summary": "string",
    "obligations": ["string"],
    "parties": ["string"],
    "flags": ["string"]
  }
]
"""
    
    # Process in batches of 3 to balance speed and memory usage
    batch_size = 3
    results = [None] * len(clauses)
    
    # Set cached results
    for idx, res in cached_results:
        results[idx] = res
    
    # Process uncached clauses in batches
    for i in range(0, len(uncached_clauses), batch_size):
        batch = uncached_clauses[i:i+batch_size]
        batch_indices = [item[0] for item in batch]
        batch_clauses = [item[1] for item in batch]
        batch_hashes = [item[2] for item in batch]
        
        prompt = "Analyze these contract clauses. For each, identify: label (clause type), risk_score (0-100), summary (plain English), obligations (list of duties), parties (involved entities), and flags (concerns). Return as a JSON array.\n\n"
        for j, clause in enumerate(batch_clauses):
            prompt += f"--- Clause {j+1} ---\n{clause}\n\n"
        
        try:
            batch_result = _ask_json(
                model, tokenizer, "Batch clause analysis", schema, prompt,
                sys_prefix="You are an expert contract lawyer. Analyze each clause independently.", 
                temp=0.3  # Lower temperature for more consistent results
            )
            
            if isinstance(batch_result, list) and len(batch_result) == len(batch):
                for idx, clause_hash, res in zip(batch_indices, batch_hashes, batch_result):
                    if "parties" not in res:
                        res["parties"] = []
                    _clause_cache[clause_hash] = res
                    results[idx] = res
            else:
                # Fallback to individual processing if batch processing fails
                for idx, clause, clause_hash in zip(batch_indices, batch_clauses, batch_hashes):
                    res = _analyze_single_clause(model, tokenizer, clause)
                    _clause_cache[clause_hash] = res
                    results[idx] = res
                    
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fallback to individual processing
            for idx, clause, clause_hash in zip(batch_indices, batch_clauses, batch_hashes):
                res = _analyze_single_clause(model, tokenizer, clause)
                _clause_cache[clause_hash] = res
                results[idx] = res
    
    return results

def _analyze_single_clause(model, tokenizer, clause):
    """Process a single clause (fallback method)."""
    schema = """
{
  "label": "string",
  "risk_score": "number",
  "summary": "string",
  "obligations": ["string"],
  "parties": ["string"],
  "flags": ["string"]
}
"""
    prompt = (
        f"Analyze this contract clause:\n\n{clause}\n\n"
        "Identify: label (clause type), risk_score (0-100), summary (plain English), "
        "obligations (list of duties/requirements), parties (list of involved parties/entities), "
        "and flags (list of concerns/warnings).\n\n"
        "Return the JSON object."
    )
    result = _ask_json(
        model, tokenizer, "Clause analysis", schema, prompt,
        sys_prefix="You are an expert contract lawyer.", temp=0.3
    )
    if "parties" not in result:
        result["parties"] = []
    return result

def process_document(model, tokenizer, files):
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract and preprocess text
    full_text = extract_text_from_files(files)
    
    # Classify document in parallel with clause splitting
    import threading
    meta_result = {}
    clauses_result = []
    
    def classify_task():
        nonlocal meta_result
        meta_result = classify_document(model, tokenizer, full_text)
    
    def split_task():
        nonlocal clauses_result
        clauses_result = split_clauses(full_text)
    
    # Run classification and splitting in parallel
    t1 = threading.Thread(target=classify_task)
    t2 = threading.Thread(target=split_task)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    meta = meta_result
    clauses = clauses_result
    
    if not clauses:
        return meta, []
    
    # Process clauses in batches
    results = []
    batch_results = analyze_clause_batch(model, tokenizer, clauses)
    
    for idx, (clause, result) in enumerate(zip(clauses, batch_results)):
        if result is None:
            # Fallback to single processing if batch failed
            result = _analyze_single_clause(model, tokenizer, clause)
        result["clause"] = clause
        result["index"] = idx + 1
        results.append(result)
    
    # Clear CUDA cache after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return meta, results
