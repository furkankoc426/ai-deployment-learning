# 2026-07-09 Environment Check

Status: GPU runtime is working in Google Colab.

Output:

```text
PyTorch version: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
CUDA version: 12.8
```

Notes:

- PyTorch build includes CUDA support.
- Colab runtime can see the Tesla T4 GPU.
- Ready for Lesson 02: PyTorch ResNet18 inference benchmark.

Next step:

Run CPU and GPU benchmark with ResNet18 for batch sizes 1, 8, and 16.
