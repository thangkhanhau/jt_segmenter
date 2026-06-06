"""
JT Segmenter — Flask Web UI
Mizo sentence segmentation served as a local web app.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import json
import uuid
import time
import shutil
import zipfile
import warnings
import unicodedata
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchcrf import CRF
from flask import (Flask, request, render_template, send_file,
                   jsonify, redirect, url_for, flash)
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — adjust paths if your checkpoint lives elsewhere
# ============================================================
APP_ROOT      = Path(__file__).parent.resolve()
CKPT_PATH     = APP_ROOT / "checkpoints" / "jt_segmenter_best.pt"
UPLOAD_DIR    = APP_ROOT / "uploads"
OUTPUT_DIR    = APP_ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME    = "xlm-roberta-base"
NUM_LABELS    = 2
DROPOUT       = 0.1
MAX_LEN       = 256
STRIDE        = 64
LABEL_PAD     = -100
INFER_BATCH   = 32
ALLOWED_EXTS  = {".txt"}
MAX_FILE_MB   = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Model definition (must match the trained architecture exactly)
# ============================================================
class JTSegmenter(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout=DROPOUT):
        super().__init__()
        self.config     = AutoConfig.from_pretrained(model_name)
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf        = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    @staticmethod
    def _pack_for_crf(emissions, labels):
        B, T, C = emissions.shape
        valid   = labels.ne(-100)
        lengths = valid.sum(dim=1)
        max_len = int(lengths.max().item())
        device  = emissions.device
        pe = torch.zeros(B, max_len, C, device=device, dtype=emissions.dtype)
        pl = torch.zeros(B, max_len,    device=device, dtype=torch.long)
        pm = torch.zeros(B, max_len,    device=device, dtype=torch.bool)
        for i in range(B):
            n = int(lengths[i].item())
            if n == 0:
                pm[i, 0] = True; continue
            sel = valid[i].nonzero(as_tuple=False).squeeze(1)
            pe[i, :n] = emissions[i, sel]
            pl[i, :n] = labels[i, sel]
            pm[i, :n] = True
        return pe, pl, pm, lengths

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden    = self.dropout(out.last_hidden_state)
        emissions = self.classifier(hidden)
        if labels is not None:
            pe, pl, pm, _ = self._pack_for_crf(emissions, labels)
            log_lik = self.crf(pe, pl, mask=pm, reduction="mean")
            return {"loss": -log_lik, "emissions": emissions}
        crf_mask = attention_mask.bool()
        decoded  = self.crf.decode(emissions, mask=crf_mask)
        return {"predictions": decoded, "emissions": emissions}


# ============================================================
# Load model + tokenizer ONCE at startup
# ============================================================
print("=" * 60)
print("JT Segmenter — Flask app starting")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
PAD_ID    = tokenizer.pad_token_id

print(f"Loading checkpoint: {CKPT_PATH}")
if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        f"Edit CKPT_PATH at the top of app.py to point at your "
        f"jt_segmenter_best.pt file."
    )
model = JTSegmenter().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"✓ Loaded epoch {ckpt['epoch']} | "
      f"val_F1 = {ckpt['best_val_f1']:.4f}")
MODEL_INFO = {
    "name":       "JT Segmenter",
    "encoder":    MODEL_NAME,
    "device":     str(DEVICE),
    "best_epoch": ckpt["epoch"],
    "val_f1":     round(float(ckpt["best_val_f1"]), 4),
    "val_p":      round(float(ckpt["val_metrics"]["precision"]), 4),
    "val_r":      round(float(ckpt["val_metrics"]["recall"]), 4),
}


# ============================================================
# Preprocessing — clean raw Mizo text into paragraphs
# ============================================================
def preprocess_text(raw_text):
    """
    Clean and normalise raw Mizo text into a list of paragraphs.

    Steps:
      1. Unicode-normalise (NFC) — important for Mizo diacritics like ṭ, ê, î
      2. Strip <p>...</p> tags if present (training-format compatibility)
      3. Treat each non-empty line as one paragraph
      4. Collapse internal whitespace runs to single spaces
      5. Drop empty / whitespace-only lines

    Returns: (paragraphs, stats_dict)
    """
    # NFC normalisation — combines combining marks into precomposed forms
    text = unicodedata.normalize("NFC", raw_text)

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    raw_lines = text.split("\n")
    paragraphs = []
    n_stripped_tags = 0
    n_dropped_empty = 0
    n_collapsed_ws  = 0

    for line in raw_lines:
        s = line.strip()
        if not s:
            n_dropped_empty += 1
            continue
        # Strip <p>...</p> wrapper if present
        if s.startswith("<p>"):
            s = s[3:]
            n_stripped_tags += 1
        if s.endswith("</p>"):
            s = s[:-4]
        s = s.strip()
        if not s:
            n_dropped_empty += 1
            continue
        # Collapse internal whitespace runs
        collapsed = " ".join(s.split())
        if collapsed != s:
            n_collapsed_ws += 1
        paragraphs.append(collapsed)

    stats = {
        "raw_lines":       len(raw_lines),
        "paragraphs_kept": len(paragraphs),
        "stripped_p_tags": n_stripped_tags,
        "dropped_empty":   n_dropped_empty,
        "collapsed_ws":    n_collapsed_ws,
        "total_tokens":    sum(len(p.split()) for p in paragraphs),
    }
    return paragraphs, stats


# ============================================================
# Encoding + Dataset (mirror of training pipeline)
# ============================================================
def encode_paragraph(tokens, tokenizer,
                     max_len=MAX_LEN, stride=STRIDE, label_pad=LABEL_PAD):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=False,
                    add_special_tokens=False)
    sub_ids  = enc["input_ids"]
    word_ids = enc.word_ids()
    sub_labels = []
    prev = None
    for w in word_ids:
        if w is None:        sub_labels.append(label_pad)
        elif w != prev:      sub_labels.append(0); prev = w   # dummy 0
        else:                sub_labels.append(label_pad)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    inner_max = max_len - 2
    if len(sub_ids) <= inner_max:
        ranges = [(0, len(sub_ids))]
    else:
        ranges, start = [], 0
        while start < len(sub_ids):
            end = min(start + inner_max, len(sub_ids))
            ranges.append((start, end))
            if end == len(sub_ids): break
            start = end - stride
    chunks = []
    for ci, (s, e) in enumerate(ranges):
        ids  = [cls_id] + sub_ids[s:e]   + [sep_id]
        labs = [label_pad] + sub_labels[s:e] + [label_pad]
        wids = [None] + word_ids[s:e]    + [None]
        chunks.append({"input_ids": ids, "attention_mask": [1]*len(ids),
                       "labels": labs, "word_ids": wids,
                       "chunk_idx": ci, "n_chunks": len(ranges)})
    return chunks


class _InferDataset(Dataset):
    def __init__(self, chunks): self.data = chunks
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def _collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    iid, am, lb, meta = [], [], [], []
    for b in batch:
        L = len(b["input_ids"]); pad = max_len - L
        iid.append(b["input_ids"]      + [PAD_ID]    * pad)
        am.append( b["attention_mask"] + [0]         * pad)
        lb.append( b["labels"]         + [LABEL_PAD] * pad)
        meta.append({"word_ids": b["word_ids"] + [None]*pad,
                     "pid": b["pid"], "n_words": b["n_words"],
                     "valid_len": L})
    return {"input_ids":      torch.tensor(iid, dtype=torch.long),
            "attention_mask": torch.tensor(am,  dtype=torch.long),
            "labels":         torch.tensor(lb,  dtype=torch.long),
            "meta": meta}


# ============================================================
# Inference — paragraphs → segmented sentences
# ============================================================
@torch.no_grad()
def segment_paragraphs(paragraphs):
    """
    Segment a list of preprocessed Mizo paragraphs.
    Returns: list[list[str]]  — outer = paragraphs, inner = sentences.
    """
    if not paragraphs:
        return []
    flat_chunks = []
    for pid, para in enumerate(paragraphs):
        toks = para.split()
        if not toks:
            continue
        chunks = encode_paragraph(toks, tokenizer)
        for ch in chunks:
            ch["pid"] = pid; ch["n_words"] = len(toks)
            flat_chunks.append(ch)

    if not flat_chunks:
        return [[] for _ in paragraphs]

    loader = DataLoader(_InferDataset(flat_chunks),
                        batch_size=INFER_BATCH, shuffle=False,
                        collate_fn=_collate)

    # word_votes[pid][word_idx] -> list[int]
    from collections import defaultdict
    word_votes     = defaultdict(lambda: defaultdict(list))
    n_words_by_pid = {}

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        preds = out["predictions"]

        for i, seq_pred in enumerate(preds):
            meta = batch["meta"][i]
            pid       = meta["pid"]
            n_words   = meta["n_words"]
            wids      = meta["word_ids"]
            valid_len = meta["valid_len"]
            n_words_by_pid[pid] = n_words
            prev_w = None
            for t in range(min(valid_len, len(seq_pred))):
                w = wids[t]
                if w is None:
                    prev_w = None; continue
                if w != prev_w:
                    word_votes[pid][w].append(seq_pred[t])
                    prev_w = w

    results = [[] for _ in paragraphs]
    for pid, para in enumerate(paragraphs):
        toks = para.split()
        n    = len(toks)
        if n == 0:
            continue
        votes = word_votes.get(pid, {})
        preds = [0] * n
        for w in range(n):
            v = votes.get(w, [])
            if v:
                ones = sum(v); zeros = len(v) - ones
                preds[w] = 1 if (ones >= zeros and ones > 0) else 0
        # Safety: ensure final token is a boundary
        if preds[-1] == 0:
            preds[-1] = 1
        if sum(preds) == 0:
            preds[-1] = 1
        # Split
        cur, sentences = [], []
        for tok, p in zip(toks, preds):
            cur.append(tok)
            if p == 1:
                sentences.append(" ".join(cur)); cur = []
        if cur:
            sentences.append(" ".join(cur))
        results[pid] = sentences
    return results


# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
app.secret_key = "jt-segmenter-local-app"


def _allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_info=MODEL_INFO,
                           max_mb=MAX_FILE_MB)


@app.route("/segment", methods=["POST"])
def segment():
    """
    Accepts one or more uploaded .txt files OR pasted text.
    Returns a JSON response with per-file results and download links.
    """
    job_id   = uuid.uuid4().hex[:12]
    job_dir  = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pasted_text = (request.form.get("pasted_text") or "").strip()
    files       = request.files.getlist("files")

    jobs = []  # list of {"name", "raw_text"}

    if pasted_text:
        jobs.append({"name": "pasted_text.txt", "raw_text": pasted_text})

    for f in files:
        if not f or not f.filename:
            continue
        if not _allowed(f.filename):
            return jsonify({"error":
                f"Unsupported file type: {f.filename}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTS))}"}), 400
        try:
            raw = f.read().decode("utf-8")
        except UnicodeDecodeError:
            return jsonify({"error":
                f"{f.filename} is not valid UTF-8. Please re-save as UTF-8."}), 400
        jobs.append({"name": secure_filename(f.filename), "raw_text": raw})

    if not jobs:
        return jsonify({"error":
            "No input received. Upload a .txt file or paste text."}), 400

    results = []
    t_total = time.time()
    for job in jobs:
        t0 = time.time()
        paragraphs, prep_stats = preprocess_text(job["raw_text"])
        if not paragraphs:
            results.append({
                "input_name": job["name"],
                "error":      "After preprocessing, no non-empty paragraphs remained.",
                "stats":      prep_stats,
            })
            continue

        segmented = segment_paragraphs(paragraphs)
        elapsed   = time.time() - t0

        # Write outputs
        base = Path(job["name"]).stem
        out_txt   = job_dir / f"{base}_segmented.txt"
        out_jsonl = job_dir / f"{base}_segmented.jsonl"

        n_sent = 0
        with open(out_txt,   "w", encoding="utf-8") as ftxt, \
             open(out_jsonl, "w", encoding="utf-8") as fjsl:
            for pid, sents in enumerate(segmented):
                for s in sents:
                    ftxt.write(s + "\n")
                ftxt.write("\n")  # blank line between paragraphs
                fjsl.write(json.dumps({
                    "pid":         pid,
                    "n_tokens":    len(paragraphs[pid].split()),
                    "n_sentences": len(sents),
                    "sentences":   sents,
                }, ensure_ascii=False) + "\n")
                n_sent += len(sents)

        # Build a small preview (first 2 paragraphs)
        preview = []
        for pid in range(min(2, len(segmented))):
            preview.append({
                "pid":       pid,
                "input":     paragraphs[pid][:300] +
                             ("..." if len(paragraphs[pid]) > 300 else ""),
                "sentences": segmented[pid],
            })

        results.append({
            "input_name":   job["name"],
            "stats":        prep_stats,
            "n_paragraphs": len(paragraphs),
            "n_sentences":  n_sent,
            "avg_per_para": round(n_sent / max(1, len(paragraphs)), 2),
            "seconds":      round(elapsed, 2),
            "txt_url":      url_for("download",
                                    job_id=job_id, fname=out_txt.name),
            "jsonl_url":    url_for("download",
                                    job_id=job_id, fname=out_jsonl.name),
            "preview":      preview,
        })

    # Bundle ZIP if multiple files
    bundle_url = None
    real_outputs = [r for r in results if "error" not in r]
    if len(real_outputs) > 1:
        zip_path = job_dir / f"jt_segmenter_{job_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in job_dir.iterdir():
                if fp.suffix in {".txt", ".jsonl"}:
                    zf.write(fp, arcname=fp.name)
        bundle_url = url_for("download", job_id=job_id, fname=zip_path.name)

    return jsonify({
        "job_id":         job_id,
        "results":        results,
        "bundle_url":     bundle_url,
        "total_seconds":  round(time.time() - t_total, 2),
    })


@app.route("/download/<job_id>/<path:fname>")
def download(job_id, fname):
    safe = secure_filename(fname)
    fp   = (OUTPUT_DIR / job_id / safe).resolve()
    # Confine to job dir
    if not str(fp).startswith(str((OUTPUT_DIR / job_id).resolve())):
        return "Forbidden", 403
    if not fp.exists():
        return "Not found", 404
    return send_file(fp, as_attachment=True, download_name=safe)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_INFO})


if __name__ == "__main__":
    print(f"\nOpen http://127.0.0.1:5000 in your browser")
    print(f"(Stop with Ctrl+C in this terminal)\n")
    # threaded=False keeps inference single-threaded → predictable VRAM use
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)