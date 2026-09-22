# AI Text Detector

A small Streamlit app for pasting or uploading one or more texts and scoring
them for AI-generated prose. It's a thin UI over the open, from-scratch
detector at [paulgp/econ-ai-detector](https://github.com/paulgp/econ-ai-detector)
(installed as a dependency, not forked) — credit and full methodology/caveats
live there.

## Setup

```bash
./setup.sh
```

This creates a venv and installs everything, including
`paulgp/econ-ai-detector` and its own dependencies (torch, transformers,
scikit-learn, pymupdf, ...) — the first run can take a few minutes and a
couple GB of disk.

Note: `paulgp/econ-ai-detector`'s `pyproject.toml` only declares the
`econ_ai_detector` Python package, so a plain `pip install` drops the
sibling `models/` directory (its LR weights + thresholds) from the wheel,
and `Detector()` can't find them. `app.py`'s `_ensure_models()` works
around this by downloading those two small files straight from the
upstream repo into the path the package expects, the first time the app
runs. No editable install or manual clone needed.

## Deploy without a terminal

No local install needed — deploy straight from GitHub in a browser:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), sign
   in (free), pick a name, choose **Streamlit** as the SDK.
2. In the new Space's **Files** tab, add this repo's `app.py` and
   `requirements.txt` (either upload them, or link the Space to this GitHub
   repo under Settings → Repository).
3. The Space builds and starts automatically; open the Space's URL in any
   browser.

[Hugging Face Spaces](https://huggingface.co/spaces) free tier gives more
RAM than Streamlit Community Cloud, which matters here since the
neural-network scorer loads a ~330MB model. If it's tight, uncheck "Run
the neural-network scorer" in the sidebar for a lighter LR-only mode.

## Run

```bash
streamlit run app.py
```

Opens in your browser. Paste text (one or more boxes) or upload multiple
`.txt`/`.pdf` files, then click **Run detection**.

- The neural-network half of the ensemble downloads ~330MB from Hugging Face
  the first time it runs, then is cached locally.
- Uncheck "Run the neural-network scorer" in the sidebar for a fast,
  no-download signal from the logistic-regression half only (no official
  flagged/not-flagged verdict, just the raw LR score).
- Results can be exported as CSV.

## What it's calibrated for

The underlying detector was trained and calibrated on economics
working-paper prose (NBER papers, published journal articles). On other
registers — abstracts, other academic fields, non-native English, general
web text — its false-positive rate is unknown. Treat a flag as fairly strong
evidence and a non-flag as weak evidence, not as a certainty either way. See
the upstream README for full details, training data, and reproducibility
notes.

## License

This app's code: MIT, matching upstream. The detector's LR weights: MIT.
The DistilRoBERTa weights (downloaded from Hugging Face): Apache-2.0.
