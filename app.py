import os, json, io
import gradio as gr
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline import extract_text_from_files, process_document, DOC_TYPES, CLAUSE_LABELS

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _vader_available = True
except Exception:
    _vader_available = False

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

theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md"
).set(
    body_background_fill="#f8f9fa",
    body_text_color="#1a202c",
    button_primary_background_fill="#ff6b35",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e2e8f0",
    button_secondary_text_color="#2d3748",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e0",
    block_background_fill="#ffffff",
    block_label_text_color="#2d3748",
    block_title_text_color="#1a202c",
)

CSS = """
/* Force light theme */
:root { 
    --cw-maxw: 1200px;
    --text-primary: #1a202c;
    --text-secondary: #2d3748;
    --text-subtle: #4a5568;
    color-scheme: light !important;
}

* {
    color-scheme: light !important;
}

body { 
    background: #f0f4f8 !important; 
    color: #1a202c !important;
}

/* Ensure all Gradio components have light background */
[class*="Component"] {
    background: #ffffff !important;
}

[class*="Component"] * {
    color: #2d3748 !important;
}

/* File upload area - make it more visible */
.file-upload, [data-testid="file-upload"], .upload-container {
    background: #ffffff !important;
    border: 2px dashed #ff6b35 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.file-upload:hover, [data-testid="file-upload"]:hover {
    background: #fff5f0 !important;
    border-color: #f7931e !important;
}

/* File preview area */
.file-preview, .file-name {
    background: #fff5f0 !important;
    color: #2d3748 !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    margin: 4px 0 !important;
}

/* Force light on all containers */
.gr-box, .gr-form, .gr-padded {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Tab content */
.tab-content, [role="tabpanel"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Row and column containers */
.gr-row, .gr-column {
    background: transparent !important;
}

.main {
    background: #f7f8fa !important;
}

/* Force light backgrounds */
.block, .form, .accordion {
    background: #ffffff !important;
    color: #1a1a1a !important;
}

/* Plot light class */
.plot-light {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

.plot-light * {
    background: #ffffff !important;
}

/* Override any dark mode */
.dark, [data-theme="dark"], [class*="dark"] {
    background: #ffffff !important;
    color: #1a1a1a !important;
}

/* Force white on everything */
body, html, #root, .app {
    background: #f7f8fa !important;
    color: #1a1a1a !important;
}

/* Plot containers - aggressive overrides */
.plot-container, .gr-plot, [class*="plot"], [class*="Plot"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Plotly specific */
.plotly, .plotly-graph-div {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

.js-plotly-plot, .plot-container.plotly {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* SVG backgrounds */
.plot-container svg, .plotly svg {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Force light theme */
:root { 
    --cw-maxw: 1200px;
    --text-primary: #1a1a1a;
    --text-secondary: #2d3748;
    --text-subtle: #4a5568;
    color-scheme: light !important;
}

* {
    color-scheme: light !important;
}

body { 
    background: #f7f8fa !important; 
    color: #1a1a1a !important;
}

/* Force all containers to light background */
.gradio-container {
    background: #f7f8fa !important;
    color: #1a1a1a !important;
}

.main {
    background: #f7f8fa !important;
}

/* Make all text darker and more visible */
label, .label, p, span, div, h1, h2, h3, h4, h5, h6 {
    color: #2d3748 !important;
}

.info {
    color: #4a5568 !important;
}

.textbox .info {
    color: #718096 !important;
}

/* Input fields */
input, textarea, select {
    background: #ffffff !important;
    color: #2d3748 !important;
    border: 1px solid #cbd5e0 !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #ff6b35 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1) !important;
}

/* Force light backgrounds */
.block, .form, .panel {
    background: #ffffff !important;
    color: #1a1a1a !important;
}

/* Tabs - force visibility */
.tabs, .tab-nav {
    background: #ffffff !important;
}

.tabs button, .tab-nav button {
    color: #2d3748 !important;
    font-weight: 600 !important;
    background: #e2e8f0 !important;
    border: 1px solid #cbd5e0 !important;
    padding: 10px 20px !important;
    margin: 2px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.tabs button:hover, .tab-nav button:hover {
    background: #cbd5e0 !important;
    color: #1a202c !important;
    transform: translateY(-1px) !important;
}

.tabs button[aria-selected="true"], .tab-nav button[aria-selected="true"] {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
    color: #ffffff !important;
    border-color: #f7931e !important;
    box-shadow: 0 4px 6px rgba(255, 107, 53, 0.2) !important;
}

.container { max-width: var(--cw-maxw) !important; margin: 0 auto !important; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(66, 153, 225, 0.08); }
.heading h1 { margin: 0 0 6px 0; font-weight: 700; letter-spacing: -0.01em; color: #1a202c; }
.subtle { color: #718096; font-size: 14px; }
.gradio-button { font-weight: 600 !important; border-radius: 10px !important; transition: all 0.2s ease !important; }
.gradio-button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }
.gradio-button.primary { background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important; color: #fff !important; border: none !important; }
.gradio-button.secondary { background: #e2e8f0 !important; color: #2d3748 !important; border: 1px solid #cbd5e0 !important; }
.gradio-textbox textarea { border-radius: 10px !important; border: 1px solid #cbd5e0 !important; }
.gradio-dataframe { border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; }
.gradio-dataframe table thead th { position: sticky; top: 0; background: #f7fafc; z-index: 1; color: #2d3748; font-weight: 600; }
blockquote { border-left: 4px solid #ff6b35; padding-left: 12px; color: #4a5568; background: #fff5f0; padding: 12px; border-radius: 4px; }

/* Chatbot specific fixes - aggressive overrides */
.chatbot, .chatbot-light, [data-testid="chatbot"], .gr-chatbot {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Force all chatbot containers to be white */
.chatbot *, .chatbot-light *, [data-testid="chatbot"] * {
    background-color: transparent !important;
}

/* Message bubbles */
.chatbot .message, .chatbot-light .message, .message-wrap, .message-row {
    background: #f7fafc !important;
    background-color: #f7fafc !important;
    color: #2d3748 !important;
    border: 1px solid #e2e8f0 !important;
    padding: 14px 16px !important;
    border-radius: 12px !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
}

/* User messages */
.chatbot .message.user, .chatbot-light .message.user, .user-row, [data-testid="user"] {
    background: linear-gradient(135deg, #fff5f0 0%, #ffe8dc 100%) !important;
    background-color: #fff5f0 !important;
    color: #2d3748 !important;
    border-color: #ffc9a8 !important;
    margin-left: 40px !important;
}

/* Bot messages */
.chatbot .message.bot, .chatbot-light .message.bot, .bot-row, [data-testid="bot"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #2d3748 !important;
    border-color: #e2e8f0 !important;
    margin-right: 40px !important;
}

/* All text in chatbot */
.chatbot .message *, .chatbot-light .message *, .message-wrap *, .message-row * {
    color: #2d3748 !important;
}

/* Chatbot container background */
.chatbot-container, .chatbot-wrapper {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Fix message container */
.message-wrap {
    background: #ffffff !important;
    color: #1a1a1a !important;
}

.message-wrap * {
    color: #1a1a1a !important;
}

/* Message rows */
.message-row {
    background: transparent !important;
}

/* User and bot message bubbles */
.user-message {
    background: linear-gradient(135deg, #fff5f0 0%, #ffe8dc 100%) !important;
    color: #2d3748 !important;
    border: 1px solid #ffc9a8 !important;
    font-weight: 500 !important;
}

.bot-message {
    background: #ffffff !important;
    color: #2d3748 !important;
    border: 1px solid #e2e8f0 !important;
}

/* Loading indicator for chat */
.chatbot-loading {
    background: #f7fafc !important;
    padding: 12px !important;
    border-radius: 8px !important;
    color: #718096 !important;
    font-style: italic !important;
}
"""

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


def run_pipeline(files):
    """Connect UI to the pipeline. pipeline.process_document returns (meta, results)."""
    try:
        print(f"[DEBUG] run_pipeline called with files: {files}")
        
        if not files:
            print("[DEBUG] No files provided, returning empty charts")
            empty_df = pd.DataFrame(columns=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"])
            # Use the proper empty chart functions
            empty_charts = [
                create_clause_distribution_chart([]),
                create_risk_score_histogram([]),
                create_risk_by_clause_chart([]),
                create_obligations_wordcloud_data([]),
                create_flags_analysis_chart([])
            ]
            return (empty_df, gr.update(visible=False), "", "{}", "", "", "") + tuple(empty_charts)

        print("[DEBUG] Processing document...")
        meta, results = process_document(model, tokenizer, files)
        print(f"[DEBUG] Got {len(results)} results")

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

        # Create visualizations with error handling
        print("[DEBUG] Creating visualizations...")
        try:
            clause_dist_chart = create_clause_distribution_chart(results)
        except Exception as e:
            print(f"[ERROR] Clause distribution chart failed: {e}")
            clause_dist_chart = create_clause_distribution_chart([])
        
        try:
            risk_hist_chart = create_risk_score_histogram(results)
        except Exception as e:
            print(f"[ERROR] Risk histogram failed: {e}")
            risk_hist_chart = create_risk_score_histogram([])
        
        try:
            risk_by_clause_chart = create_risk_by_clause_chart(results)
        except Exception as e:
            print(f"[ERROR] Risk by clause chart failed: {e}")
            risk_by_clause_chart = create_risk_by_clause_chart([])
        
        try:
            obligations_chart = create_obligations_wordcloud_data(results)
        except Exception as e:
            print(f"[ERROR] Obligations chart failed: {e}")
            obligations_chart = create_obligations_wordcloud_data([])
        
        try:
            flags_chart = create_flags_analysis_chart(results)
        except Exception as e:
            print(f"[ERROR] Flags chart failed: {e}")
            flags_chart = create_flags_analysis_chart([])

        print("[DEBUG] Pipeline completed successfully")
        return (
            df,
            gr.update(visible=True, value=card_idx, minimum=0, maximum=max(0, len(results) - 1), step=1),
            card_md,
            full_json,
            csv_text,
            meta.get("doc_type", ""),
            meta.get("doc_summary", ""),
            clause_dist_chart,
            risk_hist_chart,
            risk_by_clause_chart,
            obligations_chart,
            flags_chart
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] run_pipeline failed: {error_trace}")
        # Return empty results on error
        empty_df = pd.DataFrame(columns=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"])
        empty_charts = [
            create_clause_distribution_chart([]),
            create_risk_score_histogram([]),
            create_risk_by_clause_chart([]),
            create_obligations_wordcloud_data([]),
            create_flags_analysis_chart([])
        ]
        return (empty_df, gr.update(visible=False), f"Error: {str(e)}", "{}", "", "", "") + tuple(empty_charts)


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

def analyze_sentiment(text):
    """Analyze sentiment of user input"""
    if not _vader_available or not text:
        return {"label": "neutral", "compound": 0.0}
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": compound}

def legal_chatbot_reply(history, user_text, context_text="", temperature=0.3):
    """Chatbot that answers legal questions about documents"""
    sent = analyze_sentiment(user_text)
    
    # Build conversation context
    system_msg = """You are a helpful legal AI assistant. Answer questions about contracts and legal documents clearly and concisely. 
    If document context is provided, use it to answer questions. Be professional but friendly."""
    
    # Add document context if available
    if context_text and len(context_text.strip()) > 10:
        system_msg += f"\n\nDocument context:\n{context_text[:3000]}"
    
    # Build conversation history
    messages = [{"role": "system", "content": system_msg}]
    for msg in history[-5:]:  # Last 5 messages for context
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})
    
    # Generate response
    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
        )
        
        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        if isinstance(inputs, torch.Tensor):
            gen = output[0, inputs.shape[-1]:]
        else:
            gen = output[0, inputs['input_ids'].shape[-1]:]
        
        reply = tokenizer.decode(gen, skip_special_tokens=True).strip()
        
        # Fallback if empty
        if not reply or len(reply) < 3:
            reply = "I understand your question. Could you please provide more details or rephrase it?"
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chatbot error: {error_detail}")
        reply = f"I apologize, but I encountered an error. Please try again with a different question."
    
    return reply, sent

def create_clause_distribution_chart(results):
    """Create a pie chart showing distribution of clause types"""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="📊 Upload and analyze a document to see clause distribution",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    labels = [r.get('label', 'Unknown') for r in results]
    label_counts = Counter(labels)
    
    fig = px.pie(
        values=list(label_counts.values()),
        names=list(label_counts.keys()),
        title="📊 Clause Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(size=12, color='#1a1a1a'),
        showlegend=True,
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def create_risk_score_histogram(results):
    """Create a histogram showing risk score distribution"""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Upload and analyze a document to see risk scores",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    risk_scores = [float(r.get('risk_score', 0)) for r in results]
    
    fig = px.histogram(
        x=risk_scores,
        nbins=10,
        title="⚠️ Risk Score Distribution",
        labels={'x': 'Risk Score', 'y': 'Number of Clauses'},
        color_discrete_sequence=['#ff6b6b']
    )
    fig.update_layout(
        xaxis_title="Risk Score (0-100)",
        yaxis_title="Number of Clauses",
        font=dict(size=12, color='#1a1a1a'),
        height=400,
        margin=dict(t=50, b=50, l=50, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def create_risk_by_clause_chart(results):
    """Create a bar chart showing average risk by clause type"""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="📈 Upload and analyze a document to see risk by clause type",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    clause_risks = {}
    for r in results:
        label = r.get('label', 'Unknown')
        risk = float(r.get('risk_score', 0))
        if label not in clause_risks:
            clause_risks[label] = []
        clause_risks[label].append(risk)
    
    avg_risks = {label: np.mean(risks) for label, risks in clause_risks.items()}
    
    fig = px.bar(
        x=list(avg_risks.keys()),
        y=list(avg_risks.values()),
        title="📈 Average Risk Score by Clause Type",
        labels={'x': 'Clause Type', 'y': 'Average Risk Score'},
        color=list(avg_risks.values()),
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        xaxis_title="Clause Type",
        yaxis_title="Average Risk Score",
        font=dict(size=12, color='#1a1a1a'),
        height=400,
        margin=dict(t=50, b=100, l=50, r=20),
        xaxis_tickangle=-45,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def create_obligations_wordcloud_data(results):
    """Create data for obligations frequency analysis"""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="📋 Upload and analyze a document to see obligations",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=500,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    all_obligations = []
    for r in results:
        obligations = r.get('obligations', [])
        if obligations:
            all_obligations.extend(obligations)
    
    if not all_obligations:
        fig = go.Figure()
        fig.add_annotation(
            text="📋 No obligations found in this document",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=500,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    # Count obligation frequency
    obligation_counts = Counter(all_obligations)
    top_obligations = dict(obligation_counts.most_common(10))
    
    fig = px.bar(
        x=list(top_obligations.values()),
        y=list(top_obligations.keys()),
        orientation='h',
        title="📋 Top 10 Most Common Obligations",
        labels={'x': 'Frequency', 'y': 'Obligations'},
        color=list(top_obligations.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        font=dict(size=12, color='#1a1a1a'),
        height=500,
        margin=dict(t=50, b=50, l=200, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def create_flags_analysis_chart(results):
    """Create a chart showing flag frequency"""
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="🚩 Upload and analyze a document to see contract flags",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    all_flags = []
    for r in results:
        flags = r.get('flags', [])
        if flags:
            all_flags.extend(flags)
    
    if not all_flags:
        fig = go.Figure()
        fig.add_annotation(
            text="🚩 No flags found in this document",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280")
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        return fig
    
    flag_counts = Counter(all_flags)
    
    fig = px.treemap(
        names=list(flag_counts.keys()),
        values=list(flag_counts.values()),
        title="🚩 Contract Flags Analysis",
        color=list(flag_counts.values()),
        color_continuous_scale='Oranges'
    )
    fig.update_layout(
        font=dict(size=12, color='#1a1a1a'),
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig


with gr.Blocks(title="ClauseWise – AI Legal Document Analyzer", theme=theme, css=CSS) as demo:
    # Header
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #2e5090 100%); padding: 28px; margin: -8px -8px 16px -8px; border-radius: 0; box-shadow: 0 4px 20px rgba(255, 107, 53, 0.3);">
        <div style="max-width: 1200px; margin: 0 auto;">
            <h1 style="color: white; margin: 0 0 8px 0; font-size: 2.4em; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">⚖️ ClauseWise</h1>
            <p style="color: rgba(255,255,255,0.98); margin: 0; font-size: 1.15em; font-weight: 500;">AI-Powered Legal Document Analysis</p>
            <p style="color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 0.95em;">Powered by IBM Granite 3.2 2B | Extract clauses • Analyze risks • Visualize insights</p>
        </div>
    </div>
    """)
    
    # Disclaimer
    gr.HTML(f"""
    <div style="max-width: 1200px; margin: 0 auto 20px auto; padding: 0 16px;">
        <div style="color: #dc2626; font-size: 13px; padding: 10px 16px; background: #fef2f2; border-radius: 8px; border-left: 4px solid #dc2626;">
            {DISCLAIMER}
        </div>
    </div>
    """) 

    # Upload Section
    with gr.Column(elem_classes=["container"]):
        gr.HTML("""
        <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;">
            <h3 style="color: #1a202c; margin: 0 0 8px 0; font-size: 1.4em; font-weight: 600;">📄 Upload Your Document</h3>
            <p style="color: #718096; margin: 0 0 20px 0; font-size: 0.95em;">Upload any legal document (PDF, DOCX, or TXT) to get started</p>
        """)
        
        files = gr.File(
            label="📎 Drop files here or click to browse", 
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
            height=120
        )
        
        run_btn = gr.Button("🚀 Analyze Document", variant="primary", size="lg")
        
        gr.HTML("""
        <div style="margin-top: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px; border-left: 3px solid #3b82f6;">
            <div style="color: #1e40af; font-size: 0.95em; line-height: 1.6;">
                <strong>💡 Quick Start:</strong> Upload any legal document (contracts, NDAs, agreements) and click Analyze to extract clauses, identify risks, and generate visual insights.
            </div>
        </div>
        </div>
        """) 

    # Results Section
    with gr.Column(elem_classes=["container"]):
        gr.HTML('<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">')
        with gr.Tabs():
            with gr.Tab("📊 Visual Analytics"):
                gr.HTML("""
                <div style="background: white; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                    <h3 style="color: #1a1a1a; margin: 0 0 8px 0;">📈 Interactive Charts & Visualizations</h3>
                    <p style="color: #4a5568; margin: 0;">Explore your document through interactive charts. Hover, zoom, and click to interact!</p>
                </div>
                """)
                with gr.Row():
                    clause_dist_plot = gr.Plot(label="📊 Clause Distribution", elem_classes=["plot-light"])
                    risk_hist_plot = gr.Plot(label="⚠️ Risk Score Distribution", elem_classes=["plot-light"])
                with gr.Row():
                    risk_by_clause_plot = gr.Plot(label="📈 Risk by Clause Type", elem_classes=["plot-light"])
                    obligations_plot = gr.Plot(label="📋 Top Obligations", elem_classes=["plot-light"])
                with gr.Row():
                    flags_plot = gr.Plot(label="🚩 Contract Flags Analysis", elem_classes=["plot-light"])
                gr.HTML('</div>')
            
            with gr.Tab("📋 Table View"):
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="color: #1a1a1a; margin: 0 0 4px 0;">📊 Complete Data Table</h3>
                    <p style="color: #6b7280; margin: 0; font-size: 0.9em;">All extracted clauses with detailed analysis. Scroll to see all columns.</p>
                </div>
                """)
                table = gr.Dataframe(
                    headers=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"],
                    row_count=(1, "dynamic"), wrap=True, interactive=False
                )
                gr.HTML("""
                <div style="margin: 20px 0 12px 0;">
                    <h3 style="color: #1a1a1a; margin: 0 0 4px 0;">📦 Export Formats</h3>
                    <p style="color: #6b7280; margin: 0; font-size: 0.9em;">Copy the data or use download buttons below.</p>
                </div>
                """)
                with gr.Row():
                    json_txt = gr.Textbox(label="📦 JSON Export (Full Data)", lines=8, max_lines=20)
                    csv_txt = gr.Textbox(label="📊 CSV Export (Tabular)", lines=8, max_lines=20)

            with gr.Tab("📄 Card View"):
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="color: #1a1a1a; margin: 0 0 4px 0;">🔍 Detailed Clause Inspector</h3>
                    <p style="color: #6b7280; margin: 0; font-size: 0.9em;">Navigate through clauses one at a time.</p>
                </div>
                """)
                slider = gr.Slider(0, 0, step=1, label="🔢 Select Clause Number", visible=False)
                card = gr.Markdown("""
                📄 **No analysis yet**
                
                Upload a document and click 'Analyze Document' to see detailed clause breakdowns here.
                """)

            with gr.Tab("📝 Document Info", id="info_tab"):
                gr.HTML("""
                <div style="margin-bottom: 16px;">
                    <h3 style="color: #1a1a1a; margin: 0 0 4px 0;">📜 Document Metadata</h3>
                    <p style="color: #6b7280; margin: 0; font-size: 0.9em;">AI-generated classification and summary.</p>
                </div>
                """)
                doc_type = gr.Textbox(label="🏷️ Detected Document Type", placeholder="e.g., NDA, Service Agreement, etc.")
                doc_summary = gr.Textbox(label="📝 Executive Summary", lines=6, placeholder="AI-generated summary will appear here...")

            with gr.Tab("💬 Legal Chatbot", id="chat_tab"):
                gr.HTML("""
                <div style="background: #f0f9ff; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                    <h3 style="color: #1a1a1a; margin: 0 0 8px 0;">🤖 Ask Me Anything About Your Document</h3>
                    <p style="color: #4a5568; margin: 0;">I can answer questions about clauses, risks, obligations, and legal terms. Upload a document first for context!</p>
                </div>
                """)
                chatbot = gr.Chatbot(
                    height=450,
                    type="messages",
                    label="💬 Conversation",
                    show_copy_button=True,
                    elem_classes=["chatbot-light"]
                )
                chat_query = gr.Textbox(
                    placeholder="💡 Ask: 'What are the main risks?', 'Explain clause 5', 'Is this fair?'",
                    lines=2,
                    label="Your Question"
                )
                with gr.Row():
                    chat_temperature = gr.Slider(
                        0.0, 1.0,
                        value=0.3,
                        step=0.05,
                        label="🎯 Creativity (lower = faster & more focused)"
                    )
                with gr.Row():
                    chat_send = gr.Button("🚀 Send", variant="primary", size="lg")
                    chat_clear = gr.Button("🗑️ Clear Chat", variant="secondary", size="lg")
                sentiment_out = gr.Label(label="😊 Detected Sentiment")
                chat_state = gr.State([])

    state_json = gr.State()
    full_text_state = gr.State("")

    def run_and_store(files):
        """Run pipeline and store full text for chatbot context"""
        results = run_pipeline(files)
        # Extract full text for chatbot
        full_text = extract_text_from_files(files) if files else ""
        return results + (full_text,)
    
    run_btn.click(
        run_and_store, [files],
        [table, slider, card, json_txt, csv_txt, doc_type, doc_summary, 
         clause_dist_plot, risk_hist_plot, risk_by_clause_plot, obligations_plot, flags_plot, full_text_state]
    ).then(
        prepare_state, [json_txt], [state_json]
    )

    slider.change(on_card_change, [slider, state_json], [card])
    
    gr.HTML('</div>')  # Close results section
    
    # Export Section
    with gr.Column(elem_classes=["container"]):
        gr.HTML("""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 20px;">
            <h3 style="color: #1a1a1a; margin: 0 0 16px 0;">📥 Export & Actions</h3>
        """)
        with gr.Row():
            download_json_btn = gr.Button("📥 Download JSON", variant="secondary", size="lg")
            download_csv_btn = gr.Button("📥 Download CSV", variant="secondary", size="lg")
            clear_btn = gr.Button("🔄 Clear All", variant="secondary", size="lg")
        
        
        with gr.Row():
            json_file_output = gr.File(label="📄 JSON File", visible=False)
            csv_file_output = gr.File(label="📄 CSV File", visible=False)
        
        gr.HTML("""
        </div>
        <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 0.85em; margin-top: 20px;">
            👨‍⚖️ Made with ClauseWise | Powered by IBM Granite 3.2 2B | ⚠️ For educational purposes only
        </div>
        """)
    # End of container
    
    # Download functionality
    def download_json(json_content):
        """Create a downloadable JSON file"""
        if not json_content or json_content == "{}":
            return None
        
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
            f.write(json_content)
            temp_path = f.name
        return temp_path
    
    def download_csv(csv_content):
        """Create a downloadable CSV file"""
        if not csv_content:
            return None
        
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as f:
            f.write(csv_content)
            temp_path = f.name
        return temp_path
    
    download_json_btn.click(
        download_json,
        inputs=[json_txt],
        outputs=[json_file_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[json_file_output]
    )
    
    download_csv_btn.click(
        download_csv,
        inputs=[csv_txt],
        outputs=[csv_file_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[csv_file_output]
    )
# Clear functionality
    def clear_all():
        empty_df = pd.DataFrame(columns=["index", "clause", "label", "risk_score", "summary", "obligations", "parties", "flags"])
        empty_charts = [
            create_clause_distribution_chart([]),
            create_risk_score_histogram([]),
            create_risk_by_clause_chart([]),
            create_obligations_wordcloud_data([]),
            create_flags_analysis_chart([])
        ]
        return (
            None,
            empty_df,
            gr.update(visible=False),
            "Run the analysis to see clause details.",
            "{}",
            "",
            "",
            ""
        ) + tuple(empty_charts)
    clear_btn.click(
        clear_all,
        outputs=[files, table, slider, card, json_txt, csv_txt, doc_type, doc_summary,
                clause_dist_plot, risk_hist_plot, risk_by_clause_plot, obligations_plot, flags_plot]
    )
    
    # Chatbot functionality
    def chat_respond(chat_hist, query, full_text, temperature):
        """Handle chatbot responses"""
        if not (query and str(query).strip()):
            return (chat_hist or []), (chat_hist or []), gr.update(value=""), None
        
        try:
            history = chat_hist or []
            
            # Generate response
            reply, sent = legal_chatbot_reply(history, str(query).strip(), full_text, temperature=float(temperature))
            
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
    chat_send.click(
        chat_respond, 
        [chat_state, chat_query, full_text_state, chat_temperature], 
        [chatbot, chat_state, chat_query, sentiment_out]
    )
    chat_query.submit(
        chat_respond, 
        [chat_state, chat_query, full_text_state, chat_temperature], 
        [chatbot, chat_state, chat_query, sentiment_out]
    )
    chat_clear.click(chat_clear_fn, outputs=[chatbot, chat_state, sentiment_out])

# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="127.0.0.1",  # Use localhost for Windows
        server_port=7860,
        share=False,
        inbrowser=True  # Automatically open browser
    )
