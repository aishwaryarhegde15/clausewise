import os, json, io
import gradio as gr
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline import extract_text_from_files, process_document, DOC_TYPES

# Optional LoRA / PEFT support
PEFT = False
try:
    from peft import PeftModel
    PEFT = True
except Exception:
    pass

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "ibm-granite/granite-3.2-2b-instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "")  # optional LoRA adapter

# Choose dtype safely
if torch.cuda.is_available():
    try:
        DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        DTYPE = torch.float16
else:
    DTYPE = torch.float32

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
)

if ADAPTER_REPO and PEFT:
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_REPO)
        print("[INFO] Loaded LoRA adapter:", ADAPTER_REPO)
    except Exception as e:
        print("[WARN] Could not load adapter:", e)

DISCLAIMER = (
    "⚠️ ClauseWise is an educational tool and not legal advice. "
    "Always review outputs with a qualified lawyer."
)

def render_card(results, idx):
    """Build a single-clause Markdown card without using f-strings."""
    if not results or idx < 0 or idx >= len(results):
        return "No clause selected."

    r = results[idx]
    obligations = r.get("obligations", []) or []
    flags = r.get("flags", []) or []
    parties = r.get("parties", []) or []

    # Pieces assembled with format/concatenation to avoid f-string backslash pitfalls
    title = "### Clause {} — {}\n\n".format(r.get("index", idx + 1), r.get("label", "Unclassified"))
    risk = "**Risk score:** {}/100  \n".format(int(r.get("risk_score", 0)))
    parties_md = "**Parties:** {}\n\n".format(", ".join(parties) if parties else "-")

    clause_text = r.get("clause", "").strip()
    clause_md = "**Clause text**\n> " + clause_text + "\n\n"

    summary_md = "**Plain English**\n" + r.get("summary", "") + "\n\n"

    if obligations:
        obligations_md = "**Obligations**\n- " + "\n- ".join(obligations) + "\n\n"
    else:
        obligations_md = "**Obligations**\n—\n\n"

    if flags:
        flags_md = "**Flags**\n- " + "\n- ".join(flags)
    else:
        flags_md = "**Flags**\n—"

    return title + risk + parties_md + clause_md + summary_md + obligations_md + flags_md


def run_pipeline(files, _thinking_unused, _doc_hint_unused):
    """Connect UI to the pipeline. pipeline.process_document returns (meta, results)."""
    if not files:
        empty_df = pd.DataFrame(columns=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"])
        return (empty_df, gr.update(visible=False), "", "{}", "", "", "")

    meta, results = process_document(model, tokenizer, files)

    df = pd.DataFrame(
        results,
        columns=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"]
    )

    card_idx = 0 if results else -1
    card_md = render_card(results, card_idx) if results else "No clauses detected."

    full_json = json.dumps({"meta": meta, "clauses": results}, indent=2)
    csv_io = io.StringIO()
    df.to_csv(csv_io, index=False)
    csv_text = csv_io.getvalue()

    return (
        df,
        gr.update(visible=True, value=card_idx, minimum=0, maximum=max(0, len(results) - 1), step=1),
        card_md,
        full_json,
        csv_text,
        meta.get("doc_type", ""),
        meta.get("doc_summary", "")
    )


def on_card_change(idx, state_json):
    try:
        data = json.loads(state_json)
        return render_card(data.get("clauses", []), int(idx))
    except Exception:
        return "Unable to render clause."


def prepare_state(json_txt):
    try:
        data = json.loads(json_txt)
    except Exception:
        data = {"clauses": []}
    return json.dumps(data)


with gr.Blocks(title="ClauseWise – AI Legal Document Analyzer") as demo:
    gr.Markdown("# ClauseWise")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        enable_thinking = gr.Checkbox(label="Enable 'thinking' mode (visual only)", value=False)
        doc_hint = gr.Dropdown(choices=["(auto)"] + DOC_TYPES, value="(auto)", label="Document type hint (visual only)")

    with gr.Row():
        files = gr.File(label="Upload PDF / DOCX / TXT (multiple allowed)", file_count="multiple")
        run_btn = gr.Button("Analyze Document", variant="primary")

    with gr.Tabs():
        with gr.Tab("Table View"):
            table = gr.Dataframe(
                headers=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"],
                row_count=(1, "dynamic"), wrap=True, interactive=False
            )
            with gr.Row():
                json_txt = gr.Textbox(label="Export JSON (preview)", lines=10)
                csv_txt = gr.Textbox(label="Export CSV (preview)", lines=10)

        with gr.Tab("Card View"):
            slider = gr.Slider(0, 0, step=1, label="Clause index", visible=False)
            card = gr.Markdown("Run the analysis to see clause details.")

        with gr.Tab("Meta"):
            doc_type = gr.Textbox(label="Detected document type")
            doc_summary = gr.Textbox(label="Document summary", lines=6)

    state_json = gr.State()

    run_btn.click(
        run_pipeline, [files, enable_thinking, doc_hint],
        [table, slider, card, json_txt, csv_txt, doc_type, doc_summary]
    ).then(
        prepare_state, [json_txt], [state_json]
    )

    slider.change(on_card_change, [slider, state_json], [card])

    with gr.Row():
        gr.DownloadButton("Download JSON", variant="secondary", data=json_txt, file_name="clausewise_output.json")
        gr.DownloadButton("Download CSV", variant="secondary", data=csv_txt, file_name="clausewise_output.csv")

# IMPORTANT for Hugging Face Spaces (Docker) – expose host/port
demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860)
