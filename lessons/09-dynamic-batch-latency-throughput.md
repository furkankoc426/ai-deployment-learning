# Lesson 09 - Benchmark dynamic batches in ONNX Runtime

## Goal

Use the dynamic-batch ResNet18 ONNX model from Lesson 08 and compare batch sizes:

```text
1, 8, 16
```

For every batch size, measure:

- average batch latency
- p50 latency
- p95 latency
- average latency per image
- throughput in images per second

## The main trade-off

Batching usually improves hardware utilization, but it changes the latency/throughput balance.

```text
larger batch -> usually higher batch latency
larger batch -> often lower cost per image
larger batch -> often higher throughput
```

These are expected trends, not guarantees. Shared Colab resources and CPU/GPU transfer costs can affect the result.

## Metrics

### Batch latency

Time required for one call that contains the whole batch.

```text
batch_latency_ms
```

### Per-image latency

A simple amortized cost per image:

```text
per_image_ms = batch_latency_ms / batch_size
```

This does not mean every image independently completes at that time. All images belong to the same batch request.

### Throughput

Images processed per second:

```text
throughput = batch_size * 1000 / average_batch_latency_ms
```

Example:

```text
batch size = 8
average batch latency = 10 ms
throughput = 8 * 1000 / 10 = 800 images/sec
```

## Measurement scope

This exercise calls `session.run()` with a NumPy input and receives a NumPy output.

If CUDA is used, the measurement includes:

```text
CPU input -> GPU copy -> model inference -> CPU output copy
```

This is an end-to-end measurement from the Python application's perspective. A later lesson will use ONNX Runtime I/O Binding to keep tensors on the GPU and measure device-side inference more directly.

## Colab preparation

In a fresh runtime:

```python
!pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1
!pip install -q onnx onnxscript onnxruntime-gpu pandas
```

If ONNX Runtime was already imported before installation, restart the Colab session once.

## Colab exercise

```python
import os
import time

import numpy as np
import pandas as pd
import torch
import onnx

# Import torch before ONNX Runtime so Colab CUDA/cuDNN libraries are loaded.
import onnxruntime as ort

from torchvision.models import resnet18, ResNet18_Weights


ONNX_PATH = "resnet18_dynamic_batch.onnx"
BATCH_SIZES = [1, 8, 16]
WARMUP_RUNS = 20
MEASURE_RUNS = 100


torch.manual_seed(42)
np.random.seed(42)


# Re-create the Lesson 08 model if the Colab runtime was reset.
if not os.path.exists(ONNX_PATH):
    print("Dynamic ONNX model not found. Exporting it now...")

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    ).eval()

    example_input = torch.randn(
        1, 3, 224, 224,
        dtype=torch.float32,
    )

    batch_dim = torch.export.Dim(
        "batch",
        min=1,
        max=16,
    )

    torch.onnx.export(
        model,
        (example_input,),
        ONNX_PATH,
        opset_version=18,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            "x": {0: batch_dim}
        },
        external_data=False,
    )


onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

available_providers = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in available_providers

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if use_cuda
    else ["CPUExecutionProvider"]
)

session = ort.InferenceSession(
    ONNX_PATH,
    providers=providers,
)

input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", available_providers)
print("Session providers:", session.get_providers())
print("Input shape:", input_meta.shape)
print("Output shape:", output_meta.shape)
print("Warmup runs:", WARMUP_RUNS)
print("Measured runs:", MEASURE_RUNS)

assert not isinstance(input_meta.shape[0], int), input_meta.shape


def synchronize_device():
    if use_cuda:
        torch.cuda.synchronize()


def benchmark_batch(batch_size):
    x = np.random.randn(
        batch_size, 3, 224, 224
    ).astype(np.float32)

    feed = {input_meta.name: x}

    # Validate the output shape before benchmarking.
    output = session.run(
        [output_meta.name],
        feed,
    )[0]

    assert output.shape == (batch_size, 1000)

    # Warm up kernels, memory allocation, and graph execution.
    for _ in range(WARMUP_RUNS):
        session.run(
            [output_meta.name],
            feed,
        )

    synchronize_device()

    latencies_ms = []

    for _ in range(MEASURE_RUNS):
        synchronize_device()
        start_ns = time.perf_counter_ns()

        session.run(
            [output_meta.name],
            feed,
        )

        synchronize_device()
        end_ns = time.perf_counter_ns()

        latencies_ms.append(
            (end_ns - start_ns) / 1_000_000
        )

    avg_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))

    return {
        "batch_size": batch_size,
        "avg_batch_ms": avg_ms,
        "p50_batch_ms": p50_ms,
        "p95_batch_ms": p95_ms,
        "avg_per_image_ms": avg_ms / batch_size,
        "throughput_images_sec": batch_size * 1000 / avg_ms,
    }


results = [
    benchmark_batch(batch_size)
    for batch_size in BATCH_SIZES
]

results_df = pd.DataFrame(results)

baseline_throughput = results_df.loc[
    results_df["batch_size"] == 1,
    "throughput_images_sec",
].iloc[0]

results_df["throughput_vs_batch1"] = (
    results_df["throughput_images_sec"]
    / baseline_throughput
)

print("\nBenchmark results:")
print(
    results_df.round(3).to_string(index=False)
)

best_row = results_df.loc[
    results_df["throughput_images_sec"].idxmax()
]

print(
    "\nHighest measured throughput: "
    f"batch={int(best_row['batch_size'])}, "
    f"{best_row['throughput_images_sec']:.2f} images/sec"
)

print("PASS: dynamic-batch benchmark completed.")
```

## Reading the table

Example table structure:

```text
 batch_size  avg_batch_ms  p50_batch_ms  p95_batch_ms  avg_per_image_ms  throughput_images_sec  throughput_vs_batch1
          1         ...           ...           ...               ...                    ...                     1.0
          8         ...           ...           ...               ...                    ...                     ...
         16         ...           ...           ...               ...                    ...                     ...
```

Look for these questions:

1. Does batch latency increase as batch size increases?
2. Does average latency per image decrease?
3. Which batch size gives the highest throughput?
4. Is p95 much larger than p50?
5. Did ONNX Runtime use CUDA or CPU?

## Important interpretation

Do not select a production batch size using throughput alone.

For an interactive service, batch 16 may provide more throughput but make one user's request wait longer. For offline processing, maximizing throughput may be more important than minimizing request latency.

This creates two common deployment goals:

```text
online inference  -> latency-sensitive
offline inference -> throughput-sensitive
```

## Save your work

Save the executed notebook over:

```text
notebooks/ai_deployment_learning.ipynb
```

Suggested commit message:

```text
lesson 09 benchmark dynamic batch latency and throughput
```

## Next lesson

Use ONNX Runtime I/O Binding to keep input and output tensors on the GPU and separate model execution cost from CPU-GPU transfer cost.
