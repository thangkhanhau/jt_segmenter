# JT Segmenter
## Neural sentence segmentation for Mizo.
JT Segmenter splits running Mizo text into sentences using a fine-tuned XLM-RoBERTa encoder with a Conditional Random Field (CRF) decoding layer. This repository contains the the experiment notebook, and a small demonstration sample so other researchers can run and reproduce the pipeline.

## Installation

```bash
git clone https://github.com/<your-username>/jt-segmenter.git
cd jt-segmenter

# Conda (matches the Windows launcher, which expects an env named "mizen")
conda create -n mizen python=3.10 -y
conda activate mizen
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the matching CUDA build of PyTorch from https://pytorch.org/get-started/locally/ instead of the default CPU wheel.
