# Lesson 07 - PyTorch vs ONNX Runtime latency

## Goal

Measure batch-size-1 inference latency for the same ResNet18 model in:

1. PyTorch
2. ONNX Runtime

Lesson 06 established correctness. Now we measure performance.

Deployment rule:

```text
same model + same input + same device + same timing scope
```

## What this benchmark measures

This exercise measures simple API end-to-end latency:

```text
CPU input -> runtime inference -> CPU output
```

On GPU this includes input and output transfer overhead. This is intentional because both runtimes are measured with the same host-input/host-output scope.

Later we can use ONNX Runtime I/O Binding to keep tensors on the GPU and isolate model execution more closely.

## Colab preparation

Use a fresh Colab GPU session. Install only the GPU build of ONNX Runtime; it already contains CPU fallback support.

```python
!pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1
!pip install -q onnx onnxscript onnxruntime-gpu
```

If `onnxruntime` was already imported in the current session, restart the session after installation and then run the benchmark cell.

## Colab exercise

```python
import time
import numpy as np
import torch

# Import torch before ONNX Runtime so Colab's CUDA/cuDNN libraries are loaded.
import onnxruntime as ort

from torchvision.models import resnet18, ResNet18_Weights

WARMUP_RUNS = 20
MEASURE_RUNS = 100
ONNX_PATH = "resnet18_static.onnx"


def summarize(name, latencies_ms):
    result = {
        "runtime": name,
        "runs": len(latencies_ms),
        "avg_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }
    result["throughput_images_per_sec"] = 1000.0 / result["avg_ms"]
    return result


def benchmark(fn, sync_cuda=False):
    for _ in range(WARMUP_RUNS):
        fn()

    if sync_cuda:
        torch.cuda.synchronize()

    latencies_ms = []

    for _ in range(MEASURE_RUNS):
        if sync_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter_ns()
        fn()

        if sync_cuda:
            torch.cuda.synchronize()

        end = time.perf_counter_ns()
        latencies_ms.append((end - start) / 1_000_000)

    return latencies_ms


# Reproducible model and input.
torch.manual_seed(42)
np.random.seed(42)

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
x_cpu = torch.randn(1, 3, 224, 224, dtype=torch.float32)
x_numpy = x_cpu.numpy()

# Export while the model is still on CPU.
torch.onnx.export(
    model,
    x_cpu,
    ONNX_PATH,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    external_data=False,
)

available_providers = ort.get_available_providers()
use_cuda = (
    torch.cuda.is_available()
    and "CUDAExecutionProvider" in available_providers
)

if use_cuda:
    device = torch.device("cuda")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    device = torch.device("cpu")
    providers = ["CPUExecutionProvider"]

print("PyTorch version:", torch.__version__)
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", available_providers)
print("Benchmark device:", device)

# Both runtimes use the same device.
model = model.to(device)
session = ort.InferenceSession(ONNX_PATH, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Session providers:", session.get_providers())
print("Input name:", input_name)
print("Output name:", output_name)


def run_pytorch():
    # Start with CPU input and return CPU output, matching session.run scope.
    with torch.inference_mode():
        x_device = x_cpu.to(device)
        output = model(x_device)
        return output.cpu().numpy()


def run_onnxruntime():
    return session.run([output_name], {input_name: x_numpy})[0]


# Check correctness once before timing.
pytorch_output = run_pytorch()
onnx_output = run_onnxruntime()

np.testing.assert_allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-4,
)

sync_cuda = device.type == "cuda"

pytorch_times = benchmark(run_pytorch, sync_cuda=sync_cuda)
onnx_times = benchmark(run_onnxruntime, sync_cuda=sync_cuda)

pytorch_result = summarize("PyTorch", pytorch_times)
onnx_result = summarize("ONNX Runtime", onnx_times)

speedup = pytorch_result["avg_ms"] / onnx_result["avg_ms"]

print("\nPyTorch result:")
print(pytorch_result)

print("\nONNX Runtime result:")
print(onnx_result)

print("\nONNX Runtime speedup vs PyTorch:", round(speedup, 2), "x")

if speedup > 1:
    print("ONNX Runtime was faster in this run.")
elif speedup < 1:
    print("PyTorch was faster in this run.")
else:
    print("The average latencies were equal.")
```

## How to read the results

- `avg_ms`: average latency across all measured runs.
- `p50_ms`: median latency; half the runs were faster and half were slower.
- `p95_ms`: tail latency; 95% of runs completed at or below this value.
- `throughput_images_per_sec`: approximate images processed per second for batch size 1.
- `speedup > 1`: ONNX Runtime was faster.
- `speedup < 1`: PyTorch was faster.

Do not draw a conclusion from a single run. Colab is a shared environment, so repeat the complete measurement three times and compare whether the result is consistent.

## Small experiment

Change:

```python
MEASURE_RUNS = 100
```

to:

```python
MEASURE_RUNS = 300
```

Observe whether average and p95 latency become more stable.

## Save your work

Save the executed notebook to:

```text
notebooks/ai_deployment_learning.ipynb
```

Suggested commit message:

```text
lesson 07 benchmark pytorch vs onnx runtime latency
```

## Next lesson

Export ResNet18 with a dynamic batch dimension so the same ONNX model can accept batch sizes 1, 8, and 16.
