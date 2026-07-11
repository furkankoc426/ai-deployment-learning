# Lesson 06 - Validate PyTorch and ONNX Runtime outputs

## Goal

Before comparing speed, verify that the exported ONNX model produces nearly the same output as the original PyTorch model.

Deployment rule:

```text
correctness first -> performance second
```

A model that runs faster but changes predictions is not a successful deployment.

## What we will compare

The same deterministic input will be passed to:

1. PyTorch ResNet18
2. The exported ONNX model through ONNX Runtime

We will check:

- output shape;
- top-1 class index;
- maximum absolute difference;
- mean absolute difference;
- numerical closeness with `numpy.testing.assert_allclose`.

## Colab exercise

Run this in a fresh Colab GPU session. It also works with CPU-only ONNX Runtime.

```python
!pip install -q onnx onnxscript onnxruntime-gpu

import os
import numpy as np
import torch
import onnx
import onnxruntime as ort

from torchvision.models import resnet18, ResNet18_Weights

# Make the example reproducible.
torch.manual_seed(42)
np.random.seed(42)

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

# Create one input and reuse exactly the same values in both runtimes.
x_torch = torch.randn(1, 3, 224, 224, dtype=torch.float32)
x_numpy = x_torch.numpy()

with torch.inference_mode():
    pytorch_output = model(x_torch).numpy()

onnx_path = "resnet18_static.onnx"

torch.onnx.export(
    model,
    x_torch,
    onnx_path,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    external_data=False,
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

available_providers = ort.get_available_providers()
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers
    else ["CPUExecutionProvider"]
)

session = ort.InferenceSession(onnx_path, providers=providers)
input_name = session.get_inputs()[0].name
onnx_output = session.run(None, {input_name: x_numpy})[0]

max_abs_diff = np.max(np.abs(pytorch_output - onnx_output))
mean_abs_diff = np.mean(np.abs(pytorch_output - onnx_output))

pytorch_class = int(np.argmax(pytorch_output, axis=1)[0])
onnx_class = int(np.argmax(onnx_output, axis=1)[0])

print("ONNX file exists:", os.path.exists(onnx_path))
print("ONNX file size MB:", round(os.path.getsize(onnx_path) / 1024**2, 2))
print("Available providers:", available_providers)
print("Session providers:", session.get_providers())
print("PyTorch output shape:", pytorch_output.shape)
print("ONNX output shape:", onnx_output.shape)
print("PyTorch top-1 class:", pytorch_class)
print("ONNX top-1 class:", onnx_class)
print("Max absolute difference:", max_abs_diff)
print("Mean absolute difference:", mean_abs_diff)

assert pytorch_output.shape == onnx_output.shape
assert pytorch_class == onnx_class

np.testing.assert_allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-4,
)

print("PASS: PyTorch and ONNX Runtime outputs are numerically close.")
```

## How to read the result

A successful run should finish with:

```text
PASS: PyTorch and ONNX Runtime outputs are numerically close.
```

The two output tensors do not need to be bit-for-bit identical. Different kernels and floating-point operation ordering can create small numerical differences. The important points are that the differences remain within tolerance and the predicted class is unchanged.

## Small experiment

Change the random seed from `42` to `7` and run the cell again:

```python
torch.manual_seed(7)
np.random.seed(7)
```

The predicted class may change because the input changes, but PyTorch and ONNX Runtime should still agree with each other.

## Save your work

Save the executed notebook to:

```text
notebooks/ai_deployment_learning.ipynb
```

Suggested commit message:

```text
lesson 06 validate pytorch and onnx outputs
```

## Next lesson

Build a fair batch-size-1 latency benchmark for PyTorch and ONNX Runtime, including warmup and CUDA synchronization.
