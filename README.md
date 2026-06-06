# JT Segmenter

**The first neural sentence segmenter for the low-resource Mizo language.**

---

JT Segmenter splits running Mizo text into individual sentences. To the best of
our knowledge, it is the first neural sentence-segmentation system built for
Mizo (Lushai) — a low-resource language for which sentence boundaries are not
reliably handled by rule-based or off-the-shelf tools. It pairs a fine-tuned
[XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) encoder with a
Conditional Random Field (CRF) layer and is served through a simple local web
app, so it can be used without writing any code.

This repository contains the application, the experiment notebook (with outputs
preserved), and a small demonstration sample. The trained model is archived on
Zenodo and downloaded automatically on first run.

---

## Features

- **Mizo-aware** — trained on Mizo text, with Unicode (NFC) handling for
  characters such as `ṭ`, `ê`, and `î`.
- **Neural + CRF** — contextual XLM-RoBERTa embeddings with structured CRF
  decoding for coherent boundary prediction.
- **No-code web interface** — paste text or upload `.txt` files in the browser.
- **Handles long documents** — sliding-window inference with per-word voting,
  so no text is dropped.
- **Reproducible** — the full training and evaluation notebook is included with
  its outputs.

---

## How it works

JT Segmenter treats segmentation as binary token labeling: each word is
classified as sentence-final or not.

1. **Encoder** — `xlm-roberta-base` produces contextual subword embeddings.
2. **Classifier** — a linear layer scores the first subword of each word.
3. **CRF** — a `pytorch-crf` layer decodes the most likely label sequence.

Long paragraphs are processed with a sliding window (window length 256, stride
64); predictions from overlapping windows are resolved by majority vote per
word, and the final word of each paragraph is forced to be a boundary.

---

## Repository structure

```
jt_segmenter/
├── app.py                  # Flask web application + inference pipeline
├── start_server.bat        # One-click launcher (Windows)
├── requirements.txt
├── README.md
├── LICENSE
├── static/
│   └── style.css           # App styling
├── templates/
│   └── index.html          # App web page
├── data/                   # Small demonstration sample
└── JT Segmenter.ipynb      # Full experiment notebook (outputs included)
```

The trained model (`jt_segmenter_best.pt`) is **not** stored in this repository
because of its size; it is hosted on Zenodo (see below) and fetched
automatically.

---

## Installation

```bash
git clone https://github.com/thangkhanhau/jt_segmenter.git
cd jt_segmenter

# Conda (the Windows launcher expects an environment named "mizen")
conda create -n mizen python=3.10 -y
conda activate mizen
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the matching CUDA build of PyTorch from
<https://pytorch.org/get-started/locally/> instead of the default CPU wheel.
A GPU is optional — the app also runs on CPU.

---

## The trained model

The fine-tuned weights are archived on Zenodo:

> **DOI:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

You do not need to download it manually. On first launch the app checks for the
checkpoint and, if it is missing, downloads it from Zenodo into `checkpoints/`
automatically. Subsequent runs reuse the cached file.

The base encoder, `xlm-roberta-base`, is downloaded from the Hugging Face Hub the
first time the app or notebook runs.

---

## Usage

### Web application

**Windows:** double-click `start_server.bat`. It activates the conda environment,
starts the server, and opens your browser at <http://127.0.0.1:5000>.

**Any platform:**

```bash
conda activate mizen
python app.py
# then open http://127.0.0.1:5000
```

In the browser, paste Mizo text or upload one or more `.txt` files. The app
returns the segmented result as plain text (one sentence per line) and as JSON
Lines (one object per paragraph), with a downloadable ZIP when multiple files
are processed. Stop the server with `Ctrl+C`.

### Experiment notebook

```bash
conda activate mizen
pip install jupyter
jupyter notebook "JT Segmenter.ipynb"
```

The notebook is committed with its outputs preserved, so the reported metrics
and plots are visible without re-running. Its cells can be executed against the
sample in `data/`.

---

## Sample data

The `data/` folder contains a small Mizo sample so the app and notebook can be
tried immediately. It is provided for demonstration only.

The full corpus used to train and evaluate the model is **not redistributed**:
the source texts were collected from publicly available Mizo websites and remain
under their original copyright. Aggregate corpus statistics are provided below
for transparency.

---

## Corpus statistics

| Split       | Documents | Sentences | Tokens |
|-------------|-----------|-----------|--------|
| Training    | `<n>`     | `<n>`     | `<n>`  |
| Validation  | `<n>`     | `<n>`     | `<n>`  |
| Test        | `<n>`     | `<n>`     | `<n>`  |

- **Language:** Mizo (Lushai)
- **Sources:** publicly available Mizo websites (news, government), crawled per `robots.txt`
- **Validation performance:** F1 `<value>` · Precision `<value>` · Recall `<value>`

*(Fill in the values from your notebook outputs.)*

---

## Citation

If you use JT Segmenter in your research, please cite the model archive and the
accompanying paper:

```bibtex
@software{jtsegmenter_model,
  title     = {Jamal & Thangkhanhau Segmenter},
  author    = {Thangkhanhau et al.},
  year      = {2026},
  publisher = {},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}

@article{jtsegmenter_paper,
  title   = {Neural Sentence Segmentation for Mizo]{Neural Sentence Segmentation for Mizo: An XLM-R with CRF Approach and Empirical Analysis of Linguistic Heuristics},
  author  = {Thangkhanhau Haulai, Jamal Hussain, Dawngliani MS, Verónica Vargas Alejo},
  journal = {<Journal>},
  year    = {2026}
}
```

---

## License

- **Code and trained model:** released under the MIT License (see `LICENSE`).
- **Base encoder:** `xlm-roberta-base` is distributed under the MIT License.
- **Training corpus:** not redistributed; remains under the copyright of the
  original source websites.

---
