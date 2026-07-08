# AI Deployment Learning

This repo is for learning AI deployment step by step.

Topics:

- PyTorch inference
- latency and throughput
- ONNX export
- ONNX Runtime
- CUDA and cuDNN basics
- TensorRT
- FP32, FP16, INT8
- benchmarking
- FastAPI model serving
- Docker basics
- Triton Inference Server

The goal is not model training. The goal is to learn how to run, optimize, measure, and serve trained models.

## Learning workflow

For each topic:

1. Short theory
2. Minimal Colab code
3. Explanation
4. Small homework
5. Debug if needed
6. Save notes and results in this repo

## Main project goal

PyTorch model -> ONNX export -> ONNX Runtime CPU/GPU inference -> TensorRT engine -> benchmark -> FastAPI API -> Docker/Triton basics

## Cloud environment

Because there is no local GPU, the exercises will run on cloud notebooks:

- Google Colab
- Kaggle Notebook
- Short-term cloud GPU if needed

Start with Google Colab T4 GPU.

## Repo structure

- README.md
- roadmap.md
- progress.md
- lessons/01-inference-latency-throughput.md
- colab/01_pytorch_inference_latency.py

## First Colab check

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("No GPU")
```

First lesson: inference, latency, throughput, warmup, and benchmark basics.
