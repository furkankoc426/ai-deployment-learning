# Lesson 04 - Run ONNX model with ONNX Runtime

## Goal

Run the exported ResNet18 ONNX model with ONNX Runtime.

Flow:

```text
resnet18_static.onnx -> ONNX Runtime -> inference output
```

## Why this matters

ONNX Runtime lets us run an ONNX model without using PyTorch for inference.

This is important for deployment because the model becomes less tied to the training framework.

## Install packages

In Colab:

```python
!pip install onnxruntime onnxruntime-gpu
```

## Important concepts

### InferenceSession

This loads the ONNX model:

```python
session = ort.InferenceSession("resnet18_static.onnx")
```

### Providers

Providers decide where the model runs.

Common providers:

```text
CPUExecutionProvider
CUDAExecutionProvider
```

For GPU:

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

The CPU provider is used as fallback if needed.

### Input and output names

We exported the model with:

```text
input name: input
output name: output
```

ONNX Runtime needs inputs as NumPy arrays.

## Colab exercise

```python
!pip install onnxruntime onnxruntime-gpu

import time
import statistics
import numpy as np
import onnxruntime as ort

onnx_path = "resnet18_static.onnx"

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

session = ort.InferenceSession(
    onnx_path,
    providers=providers,
)

print("Session providers:", session.get_providers())

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input name:", input_name)
print("Output name:", output_name)
print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warmup
for _ in range(10):
    session.run([output_name], {input_name: x})

latencies_ms = []
for _ in range(50):
    start = time.time()
    outputs = session.run([output_name], {input_name: x})
    end = time.time()
    latencies_ms.append((end - start) * 1000)

avg_latency = statistics.mean(latencies_ms)
p50_latency = statistics.median(latencies_ms)
p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1]
throughput = 1000.0 / avg_latency

print("Output type:", type(outputs[0]))
print("Output shape:", outputs[0].shape)
print("Avg latency ms:", avg_latency)
print("P50 latency ms:", p50_latency)
print("P95 latency ms:", p95_latency)
print("Throughput images/sec:", throughput)
```

## Expected result

You should see:

```text
Available providers: ... CUDAExecutionProvider ... CPUExecutionProvider ...
Session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Input name: input
Output name: output
Output shape: (1, 1000)
```

## Homework

Run the code and save the notebook to GitHub again.

We will compare:

- PyTorch GPU latency
- ONNX Runtime GPU latency

Next lesson:

ONNX Runtime CPU vs GPU benchmark with batch sizes 1, 8, and 16.
