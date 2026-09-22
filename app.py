"""Streamlit UI over paulgp/econ-ai-detector: paste or upload one or more
texts (.txt/.pdf) and score each for AI-generated prose.

Run with: streamlit run app.py
"""
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Text Detector", page_icon="🔍", layout="wide")

# A plain `pip install` of econ-ai-detector doesn't ship its models/ folder
# (its pyproject.toml only declares the econ_ai_detector package, dropping
# the sibling models/ directory from the wheel), so Detector() can't find
# its weights. Fetch the two small files it needs into a scratch directory
# and point the package at that instead of its own install location, which
# on some hosts (e.g. Streamlit Community Cloud) is read-only at runtime.
_MODELS_RAW = "https://raw.githubusercontent.com/paulgp/econ-ai-detector/main/models"


def _ensure_models():
    import econ_ai_detector.detector as detector_mod

    models_dir = Path(tempfile.gettempdir()) / "econ_ai_detector_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("lr_v3.joblib", "thresholds.json"):
        dest = models_dir / fname
        if not dest.exists():
            urllib.request.urlretrieve(f"{_MODELS_RAW}/{fname}", dest)
    detector_mod.MODELS = models_dir


def k2(values):
    """2nd-highest value, or a sentinel if there's only one window."""
    s = sorted(values, reverse=True)
    return float(s[1]) if len(s) > 1 else -1e9


@st.cache_resource(show_spinner=False)
def load_detector(target_fpr: str):
    _ensure_models()
    from econ_ai_detector import Detector
    return Detector(target_fpr=target_fpr)


def score_item(det, name: str, text: str = None, pdf_path: str = None,
               gate: bool = True, use_nn: bool = True):
    from econ_ai_detector.preprocess import cut_references, pdf_text, prose_ok, windows

    raw = pdf_text(pdf_path) if pdf_path else text
    raw = cut_references(raw) if pdf_path else raw
    wins = windows(raw)
    kept = [w for w in wins if prose_ok(w)] if gate else wins

    lr = det.lr_scores(kept)
    lr_k2 = k2(lr)
    row = {
        "name": name, "n_windows": len(wins), "n_scored": len(kept),
        "lr_k2": lr_k2, "lr_threshold": det.thr["lr_k2_min"],
        "nn_k2": None, "nn_threshold": None, "flagged": None, "note": "",
    }

    if not use_nn:
        row["note"] = "LR-only: no official flagged/not-flagged verdict (needs NN too)"
        return row, lr, []

    try:
        nn = det.nn_margins(kept)
    except Exception as e:  # e.g. no network to fetch the HF weights
        row["note"] = f"NN scorer unavailable ({e}); showing LR-only signal"
        return row, lr, []

    nn_k2 = k2(nn)
    row["nn_k2"] = nn_k2
    row["nn_threshold"] = det.thr["nn_margin_k2_min"]
    row["flagged"] = bool(lr_k2 >= det.thr["lr_k2_min"] and nn_k2 >= det.thr["nn_margin_k2_min"])
    if len(kept) < 2:
        row["note"] = "fewer than 2 scored windows: the flag rule can't trigger reliably"
    return row, lr, nn


st.title("🔍 AI Text Detector")
st.caption(
    "Wraps the open, from-scratch ensemble from "
    "[paulgp/econ-ai-detector](https://github.com/paulgp/econ-ai-detector) "
    "(LR + DistilRoBERTa, calibrated on economics working-paper prose)."
)

with st.expander("What this is and isn't"):
    st.markdown(
        "- **Calibrated on economics working-paper prose.** On other registers "
        "(abstracts, other fields, non-native prose) the false-positive rate is unknown.\n"
        "- **Window resolution is 250 words.** Short texts (< 2 windows) can't trip the "
        "official flag rule; you'll still see per-window scores.\n"
        "- **A flag is strong evidence, a non-flag is weak evidence** — it's a conservative "
        "instrument, not a certainty.\n"
        "- The neural-network scorer downloads ~330MB from Hugging Face the first time "
        "it runs in this environment; after that it's cached.\n"
        "- Code: MIT. DistilRoBERTa weights: Apache-2.0 (Hugging Face)."
    )

with st.sidebar:
    st.header("Settings")
    target_fpr = st.selectbox(
        "Operating point (paper-level false-positive rate)",
        ["0.001", "0.005", "0.01"], index=0,
        help="Lower = fewer false positives, but flags less AI text.",
    )
    gate = st.checkbox(
        "Apply prose gate", value=True,
        help="Drops front matter, tables, math and reference fragments before scoring. "
             "Recommended, especially for PDFs.",
    )
    use_nn = st.checkbox(
        "Run the neural-network scorer", value=True,
        help="Needed for the official flagged/not-flagged verdict. Uncheck for a fast, "
             "no-download LR-only signal.",
    )

st.subheader("1. Give it text")
tab_paste, tab_upload = st.tabs(["Paste text", "Upload files"])

pasted_items = []
with tab_paste:
    n_paste = st.number_input("How many texts to paste?", min_value=1, max_value=10, value=1)
    for i in range(int(n_paste)):
        c1, c2 = st.columns([1, 3])
        with c1:
            label = st.text_input(f"Name #{i + 1}", value=f"pasted-{i + 1}", key=f"name_{i}")
        with c2:
            body = st.text_area(f"Text #{i + 1}", height=120, key=f"body_{i}")
        if body.strip():
            pasted_items.append((label or f"pasted-{i + 1}", body))

uploaded_files = []
with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload one or more .txt or .pdf files", type=["txt", "pdf"],
        accept_multiple_files=True,
    )

run = st.button("2. Run detection", type="primary")

if run:
    n_items = len(pasted_items) + len(uploaded_files or [])
    if n_items == 0:
        st.warning("Paste some text or upload at least one file first.")
        st.stop()

    with st.spinner("Loading detector (first run downloads model weights)..."):
        det = load_detector(target_fpr)

    rows, detail = [], {}
    progress = st.progress(0.0)
    done = 0

    for name, text in pasted_items:
        row, lr, nn = score_item(det, name, text=text, gate=gate, use_nn=use_nn)
        rows.append(row)
        detail[name] = (lr, nn)
        done += 1
        progress.progress(done / n_items)

    for f in uploaded_files or []:
        suffix = Path(f.name).suffix.lower()
        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            row, lr, nn = score_item(det, f.name, pdf_path=tmp_path, gate=gate, use_nn=use_nn)
        else:
            text = f.read().decode("utf-8", errors="replace")
            row, lr, nn = score_item(det, f.name, text=text, gate=gate, use_nn=use_nn)
        rows.append(row)
        detail[f.name] = (lr, nn)
        done += 1
        progress.progress(done / n_items)

    st.subheader("Results")
    df = pd.DataFrame(rows)
    display_df = df.copy()
    display_df["flagged"] = display_df["flagged"].map(
        {True: "🚩 flagged", False: "not flagged", None: "—"}
    )
    st.dataframe(display_df, width="stretch")

    st.download_button(
        "Download results as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name="ai_text_detector_results.csv", mime="text/csv",
    )

    for row in rows:
        lr, nn = detail[row["name"]]
        with st.expander(f"Per-window scores: {row['name']}"):
            if row["note"]:
                st.info(row["note"])
            win_df = pd.DataFrame({
                "window": range(1, len(lr) + 1),
                "lr_prob_AI": lr,
                **({"nn_margin": nn} if len(nn) == len(lr) else {}),
            })
            st.dataframe(win_df, width="stretch")
