# Lesson 05 - Fix and verify the ONNX Runtime CUDA provider

## Goal

Install only the GPU build of ONNX Runtime and verify that inference really runs through the CUDA execution provider.

## Key idea

These two packages expose the same Python module:

```text
onnxruntime      -> CPU build
onnxruntime-gpu  -> GPU build, including CPU fallback
```

Only one should be installed in an environment. Installing both can leave the CPU build active, which is why `CUDAExecutionProvider` was missing.

## Part 1 - Clean installation

Run this cell once:

```python
!pip uninstall -y onnxruntime onnxruntime-gpu
!pip install -q onnxruntime-gpu
```

Then use:

```text
Runtime -> Restart session
```

Restarting matters because Python may already have loaded the old ONNX Runtime binary.

## Part 2 - Verify the provider

After the restart, run this cell first:

```python
import torch
import onnxruntime as ort

print("PyTorch CUDA:", torch.version.cuda)
print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())

assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "CUDAExecutionProvider is still missing"
)
```

Importing PyTorch first preloads the CUDA and cuDNN libraries that its CUDA build supplies.

Expected result:

```text
Available providers: [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Part 3 - Recreate the temporary ONNX artifact

A Colab restart removes runtime files, so export the model again:

```python
import os
import torch
import onnx
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
sample = torch.randn(1, 3, 224, 224)
onnx_path = "resnet18_static.onnx"

torch.onnx.export(
    model,
    sample,
    onnx_path,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    external_data=False,
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("Model exists:", os.path.exists(onnx_path))
print("Model size MB:", round(os.path.getsize(onnx_path) / 1024**2, 2))
```

`external_data=False` keeps the weights in one ONNX file. The earlier 0.08 MB file was only the small graph file; newer PyTorch exporters can put weights in a separate sidecar file.

## Part 4 - One CUDA inference

```python
import numpy as np

session = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Session providers:", session.get_providers())

input_name = session.get_inputs()[0].name
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: x})[0]

print("Output shape:", output.shape)
print("Predicted class index:", int(output.argmax(axis=1)[0]))

assert session.get_providers()[0] == "CUDAExecutionProvider"
assert output.shape == (1, 1000)
```

## What success means

If both assertions pass:

- the GPU build is installed;
- the CUDA provider is available;
- the session prioritizes CUDA;
- the ONNX graph completes an inference.

Save the executed notebook to the same GitHub path:

```text
notebooks/ai_deployment_learning.ipynb
```

Suggested commit message:

```text
lesson 05 verify onnx runtime cuda provider
```

Next lesson: fair PyTorch vs ONNX Runtime GPU benchmarking with synchronization and batch sizes 1, 8, and 16.
