# Progress Log

Use this file to track daily learning progress.

## Current status

Current phase: Phase 1 - Inference basics
Current lesson: Lesson 01 - Inference, latency, throughput
Environment: Google Colab with GPU

## Daily log

### 2026-07-08

Created the repo and added the initial learning plan.

Next action:

Run the first Colab GPU check:

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

Expected result:

- CUDA available should be True
- GPU should be visible, for example Tesla T4, L4, or A100

## Results table

| Date | Topic | Environment | Result | Notes |
| ---- | ----- | ----------- | ------ | ----- |
| 2026-07-08 | Repo setup | GitHub | Done | Initial roadmap added |

## Questions to answer later

- What GPU did Colab provide?
- What PyTorch version is installed?
- What CUDA version is used by PyTorch?
- Is GPU inference faster than CPU inference for batch size 1?
- How does batch size change throughput?
