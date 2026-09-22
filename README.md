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

This creates a venv, installs the app's own deps, then clones
`paulgp/econ-ai-detector` into `vendor/` and installs it **in editable
mode**. That's a deliberate workaround, not the documented install: a plain
`pip install git+https://github.com/paulgp/econ-ai-detector` currently
crashes on first use, because its `pyproject.toml` only declares the
`econ_ai_detector` Python package and drops the sibling `models/` directory
(the LR weights + thresholds) from the installed wheel — `Detector()` then
can't find `models/lr_v3.joblib`. An editable install keeps the package
pointed at the cloned source tree, where `models/` is still sitting next to
it, so it works. If upstream fixes their packaging, this can go back to a
normal `pip install` line in `requirements.txt`.

Installing pulls in the detector's own dependencies (torch, transformers,
scikit-learn, pymupdf, ...), so the first run of `setup.sh` can take a few
minutes and a couple GB of disk.

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
