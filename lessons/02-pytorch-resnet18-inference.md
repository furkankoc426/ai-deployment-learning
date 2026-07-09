# Lesson 02 - PyTorch ResNet18 Inference Benchmark

## Goal

In Lesson 01, the focus was the meaning of inference, latency, throughput, warmup, and GPU synchronization.

In this lesson, the goal is to run a real pretrained model and measure CPU vs GPU inference.

Model: ResNet18
Input shape: batch_size x 3 x 224 x 224
Environment: Google Colab with GPU enabled

## Key ideas

### 1. eval mode

Use eval mode for inference.

```python
model.eval()
```

This disables training behavior in layers such as dropout and batch normalization.

### 2. no_grad

Use no_grad during inference.

```python
with torch.no_grad():
    output = model(x)
```

This prevents PyTorch from storing gradient information and reduces memory usage.

### 3. warmup

Do not measure the first few runs. GPU setup and kernel selection can make the first runs slower.

```python
for _ in range(10):
    model(x)
```

### 4. synchronize for GPU timing

CUDA work can be asynchronous, so synchronize before and after timing.

```python
torch.cuda.synchronize()
start = time.time()
model(x)
torch.cuda.synchronize()
end = time.time()
```

## Colab exercise

Step 1: Open Google Colab.

Step 2: Enable GPU.

Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU

Step 3: Run the code below.

```python
import time
import statistics

import torch
from torchvision.models import resnet18, ResNet18_Weights

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

def benchmark(device_name, batch_size, warmup_iters=10, measure_iters=30):
    device = torch.device(device_name)
    model.to(device)
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for _ in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()
            times.append((end - start) * 1000)

    avg_ms = statistics.mean(times)
    p50_ms = statistics.median(times)
    p95_ms = sorted(times)[int(len(times) * 0.95) - 1]
    throughput = batch_size * 1000 / avg_ms

    return {
        "device": device_name,
        "batch_size": batch_size,
        "avg_latency_ms": round(avg_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_images_per_second": round(throughput, 2),
    }

for batch_size in [1, 8, 16]:
    print(benchmark("cpu", batch_size))

    if torch.cuda.is_available():
        print(benchmark("cuda", batch_size))
```

## What to observe

After running the code, compare these points:

1. CPU latency for batch size 1
2. GPU latency for batch size 1
3. CPU throughput for batch size 16
4. GPU throughput for batch size 16
5. Whether bigger batch size improves throughput
6. Whether bigger batch size increases single-request latency

## Expected learning

After this exercise, you should understand this deployment idea:

A GPU is not automatically faster for every single tiny request, but it usually gives much better throughput when batching is used.

## Save your result

Copy the printed output into progress.md or a new log file.

Use this format:

```text
Date:
Environment:
GPU:
PyTorch version:
CUDA version:
Batch size 1 CPU:
Batch size 1 GPU:
Batch size 16 CPU:
Batch size 16 GPU:
Short note:
```

## Next lesson

Lesson 03 will use the same model and export it from PyTorch to ONNX.
