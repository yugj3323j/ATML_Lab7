import os
import pandas as pd
import streamlit as st
import torch

from utils import load_metrics, translate_sentence, translate_sentence_pretrained

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
LOSS_PLOT_PATH = os.path.join(MODELS_DIR, "training_loss.png")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
LEGACY_ATTENTION_MODEL_PATH = os.path.join(MODELS_DIR, "attention_model.pt")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap');

:root {
    --bg:        #06070d;
    --bg2:       #0d0f1c;
    --surface:   rgba(255,255,255,0.03);
    --border:    rgba(255,255,255,0.07);
    --neon-g:    #00ffb3;
    --neon-b:    #3d7fff;
    --neon-p:    #bf5fff;
    --neon-y:    #ffd166;
    --text:      #dde2f0;
    --muted:     #5a607a;
    --glow-g:    rgba(0,255,179,0.18);
    --glow-b:    rgba(61,127,255,0.18);
    --glow-p:    rgba(191,95,255,0.18);
}

/* ── Reset / base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* scanning-line animation on body */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        transparent 0px,
        transparent 3px,
        rgba(0,255,179,0.012) 3px,
        rgba(0,255,179,0.012) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ambient orbs */
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    width: 700px; height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(61,127,255,0.12) 0%, transparent 70%);
    top: -200px; right: -200px;
    pointer-events: none;
    z-index: 0;
    animation: drift 20s ease-in-out infinite alternate;
}
@keyframes drift {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(-60px, 80px) scale(1.1); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header strip ── */
.top-strip {
    background: linear-gradient(90deg, rgba(0,255,179,0.06), rgba(61,127,255,0.06));
    border-bottom: 1px solid var(--border);
    border-radius: 0 0 16px 16px;
    padding: 2.4rem 2rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.top-strip::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-g), var(--neon-b), var(--neon-p));
}
.brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--neon-g);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-tag::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--neon-g);
    box-shadow: 0 0 8px var(--neon-g);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #fff 20%, var(--neon-g) 60%, var(--neon-b) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.sub-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.06em;
}

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--neon-b);
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--neon-b), transparent);
}

/* ── Glass panel ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: relative;
    z-index: 1;
    margin-bottom: 1rem;
}
.panel-accent-g { border-top: 2px solid var(--neon-g); }
.panel-accent-b { border-top: 2px solid var(--neon-b); }
.panel-accent-p { border-top: 2px solid var(--neon-p); }

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform .22s ease, box-shadow .22s ease;
    z-index: 1;
}
.stat-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0;
    transition: opacity .22s ease;
}
.stat-card:hover { transform: translateY(-5px); }
.stat-card:hover::before { opacity: 1; }

.stat-card.green::before { background: var(--glow-g); }
.stat-card.blue::before  { background: var(--glow-b); }
.stat-card.purple::before { background: var(--glow-p); }
.stat-card.green { box-shadow: 0 0 0 1px rgba(0,255,179,0.1); }
.stat-card.blue  { box-shadow: 0 0 0 1px rgba(61,127,255,0.1); }
.stat-card.purple { box-shadow: 0 0 0 1px rgba(191,95,255,0.1); }
.stat-card:hover.green { box-shadow: 0 8px 30px rgba(0,255,179,0.2); }
.stat-card:hover.blue  { box-shadow: 0 8px 30px rgba(61,127,255,0.2); }
.stat-card:hover.purple { box-shadow: 0 8px 30px rgba(191,95,255,0.2); }

.stat-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
}
.stat-val.green  { color: var(--neon-g); text-shadow: 0 0 20px rgba(0,255,179,0.4); }
.stat-val.blue   { color: var(--neon-b); text-shadow: 0 0 20px rgba(61,127,255,0.4); }
.stat-val.purple { color: var(--neon-p); text-shadow: 0 0 20px rgba(191,95,255,0.4); }

/* ── Translation output box ── */
.out-box {
    background: rgba(0,255,179,0.04);
    border: 1px solid rgba(0,255,179,0.2);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.out-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-g), var(--neon-b));
}
.out-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--neon-g);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.out-tag::before {
    content: '▶';
    font-size: 0.5rem;
    color: var(--neon-g);
}
.out-text {
    font-size: 1.5rem;
    font-weight: 600;
    color: #fff;
    line-height: 1.6;
    letter-spacing: 0.01em;
}

/* ── Config badge strip ── */
.badge-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    padding: 0.28rem 0.7rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: rgba(255,255,255,0.03);
}
.badge span { color: var(--neon-y); }

/* ── Input field ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #fff !important;
    padding: 0.85rem 1.1rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color .25s, box-shadow .25s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--neon-g) !important;
    box-shadow: 0 0 0 3px rgba(0,255,179,0.1), 0 0 24px rgba(0,255,179,0.1) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00ffb3 0%, #3d7fff 100%) !important;
    color: #06070d !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    transition: transform .2s, box-shadow .2s !important;
    box-shadow: 0 4px 20px rgba(0,255,179,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,255,179,0.35) !important;
}

/* ── Sidebar model selector ── */
.model-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--neon-g);
    background: rgba(0,255,179,0.07);
    border: 1px solid rgba(0,255,179,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    letter-spacing: 0.04em;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* ── Image ── */
[data-testid="stImage"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Radio ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem !important;
    color: var(--text) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    margin-top: 3rem;
    padding: 1.2rem 0;
    border-top: 1px solid var(--border);
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--neon-g) !important; }

/* sidebar logo block */
.sidebar-logo {
    padding: 1.4rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
    text-align: center;
}
.sidebar-logo .logo-icon {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 0.3rem;
}
.sidebar-logo .logo-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--neon-g), var(--neon-b));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
</style>
"""


@st.cache_data
def get_metrics(metrics_mtime: float):
    return load_metrics(METRICS_PATH)


def normalize_metrics(metrics):
    if "lstm" in metrics:
        return metrics
    if "attention" in metrics:
        metrics["lstm"] = metrics["attention"]
        return metrics
    raise ValueError("Expected `lstm` or legacy `attention` metrics in metrics.json.")


def main():
    st.set_page_config(
        page_title="NMT · En→Hi Translator · Yug Nagda",
        page_icon="⚡",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">'
            '<span class="logo-icon">⚡</span>'
            '<div class="logo-name">NMT Lab</div>'
            '<div class="logo-sub">ATML · Lab 7 · I050</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
            'letter-spacing:0.18em;text-transform:uppercase;color:#3d7fff;'
            'margin-bottom:0.6rem;">// Select Model</p>',
            unsafe_allow_html=True,
        )

        best_label  = "Encoder–Decoder with Attention (best)"
        local_label = "Local LSTM + Attention (checkpoint)"
        model_choice = st.radio(
            "model",
            options=[best_label, local_label],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="model-badge">architecture: seq2seq</div>'
            '<div class="model-badge">attention: bahdanau</div>'
            '<div class="model-badge">rnn: bi-lstm</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#5a607a;font-size:0.7rem;font-family:\'JetBrains Mono\','
            'monospace;text-align:center;letter-spacing:0.08em;">'
            'Yug Nagda · I050<br>ATML Lab 7</p>',
            unsafe_allow_html=True,
        )

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="top-strip">'
        '<div class="brand-tag">Neural Machine Translation · Live Demo</div>'
        '<div class="main-title">English → Hindi</div>'
        '<div class="sub-title">encoder-decoder · bahdanau attention · lstm · pytorch</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    has_lstm    = os.path.exists(LSTM_MODEL_PATH) or os.path.exists(LEGACY_ATTENTION_MODEL_PATH)
    has_metrics = os.path.exists(METRICS_PATH)

    if not (has_lstm or has_metrics):
        st.warning(
            "Training artifacts not found.\n\n"
            "Run `python train.py --data_path <dataset>` first.\n\n"
            f"Expected: `{os.path.relpath(METRICS_PATH, PROJECT_DIR)}`"
        )
        if st.button("⟳  Reload"):
            st.rerun()

    # ── Translation Panel ─────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">// Translation Terminal</div>',
        unsafe_allow_html=True,
    )

    col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_in:
        input_text = st.text_input(
            "input",
            placeholder="Type an English sentence…",
            label_visibility="collapsed",
        )
    with col_btn:
        translate_clicked = st.button("Translate ⚡", use_container_width=True)

    if translate_clicked:
        if model_choice == local_label and not has_lstm:
            st.info("Local checkpoint not available yet. Reload after training.")
        elif not input_text.strip():
            st.warning("Please enter an English sentence.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with st.spinner("Running inference…"):
                if model_choice == best_label:
                    result = translate_sentence_pretrained(sentence=input_text, device=device)
                else:
                    result = translate_sentence(
                        sentence=input_text,
                        models_dir=MODELS_DIR,
                        device=device,
                    )
            translation = result["translation"] if result["translation"] else "(no translation)"
            st.markdown(
                '<div class="out-box">'
                '<div class="out-tag">Hindi Output</div>'
                f'<div class="out-text">{translation}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Metrics ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">// Model Performance</div>',
        unsafe_allow_html=True,
    )

    if has_metrics:
        metrics_mtime = os.path.getmtime(METRICS_PATH)
        metrics = normalize_metrics(get_metrics(metrics_mtime))
        bleu       = round(metrics["lstm"]["bleu_score"], 4)
        train_loss = round(metrics["lstm"]["final_train_loss"], 4)
        val_loss   = round(metrics["lstm"]["final_val_loss"], 4)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="stat-card green">'
                '<div class="stat-icon">🎯</div>'
                '<div class="stat-lbl">BLEU Score</div>'
                f'<div class="stat-val green">{bleu}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="stat-card blue">'
                '<div class="stat-icon">📉</div>'
                '<div class="stat-lbl">Train Loss</div>'
                f'<div class="stat-val blue">{train_loss}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="stat-card purple">'
                '<div class="stat-icon">🔍</div>'
                '<div class="stat-lbl">Val Loss</div>'
                f'<div class="stat-val purple">{val_loss}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Config panel
        if "config" in metrics:
            cfg = metrics["config"]
            st.markdown(
                '<div class="sec-label">// Hyperparameter Config</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel panel-accent-b">'
                '<div class="badge-strip">'
                f'<span class="badge">epochs: <span>{cfg.get("epochs","–")}</span></span>'
                f'<span class="badge">batch: <span>{cfg.get("batch_size","–")}</span></span>'
                f'<span class="badge">embed_dim: <span>{cfg.get("embedding_dim","–")}</span></span>'
                f'<span class="badge">hidden_dim: <span>{cfg.get("hidden_dim","–")}</span></span>'
                f'<span class="badge">lr: <span>{cfg.get("learning_rate","–")}</span></span>'
                f'<span class="badge">dropout: <span>{cfg.get("dropout","–")}</span></span>'
                f'<span class="badge">tf_ratio: <span>{cfg.get("teacher_forcing_ratio","–")}</span></span>'
                f'<span class="badge">layers: <span>{cfg.get("num_layers","–")}</span></span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Loss curve
        if os.path.exists(LOSS_PLOT_PATH):
            st.markdown(
                '<div class="sec-label">// Training Loss Curve</div>',
                unsafe_allow_html=True,
            )
            st.image(LOSS_PLOT_PATH, caption="Loss per epoch", use_container_width=True)

        # Sample translations
        st.markdown(
            '<div class="sec-label">// Sample Translations</div>',
            unsafe_allow_html=True,
        )
        rows = [
            {
                "English":      r["english"],
                "Reference":    r["reference"],
                "LSTM Output":  r["prediction"],
            }
            for r in metrics["lstm"]["sample_translations"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

    else:
        st.info("Metrics will appear after training writes `saved_models/metrics.json`.")

    # ── Dataset info ──────────────────────────────────────────────────────
    if has_metrics and "dataset" in metrics:
        ds = metrics["dataset"]
        st.markdown(
            '<div class="sec-label">// Dataset Overview</div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(
                '<div class="stat-card green">'
                '<div class="stat-lbl">Total Samples</div>'
                f'<div class="stat-val green" style="font-size:1.5rem;">'
                f'{ds.get("total_samples","–"):,}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                '<div class="stat-card blue">'
                '<div class="stat-lbl">Train Samples</div>'
                f'<div class="stat-val blue" style="font-size:1.5rem;">'
                f'{ds.get("train_samples","–"):,}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                '<div class="stat-card purple">'
                '<div class="stat-lbl">Test Samples</div>'
                f'<div class="stat-val purple" style="font-size:1.5rem;">'
                f'{ds.get("test_samples","–"):,}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="footer">⚡ NMT Dashboard · Yug Nagda · I050 · ATML Lab 7 · '
        'Encoder–Decoder with Bahdanau Attention · PyTorch</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
