# 2026-07-09 ONNX Export

Environment:

```text
PyTorch version: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
CUDA version: 12.8
```

Goal:

Export torchvision ResNet18 from PyTorch to ONNX.

Output file:

```text
resnet18_static.onnx
```

## Issue found

First attempt failed because the package `onnxscript` was missing.

Error:

```text
ModuleNotFoundError: No module named 'onnxscript'
```

Fix:

```python
!pip install onnx onnxscript
```

## Export result

Export succeeded after installing `onnxscript`.

```text
ONNX export successful: resnet18_static.onnx
Input name: input
Output name: output
Number of nodes: 49
File size MB: 0.08
```

## Warning observed

PyTorch 2.11 warned that it prefers opset 18 even though opset 17 was requested.

The exporter tried to convert the model to opset 17, but conversion failed. The final model was still exported successfully.

Learning note:

For PyTorch 2.11, prefer `opset_version=18` for this model unless there is a runtime compatibility reason to use an older opset.

## Key learning

- ONNX export needs a dummy input.
- Recent PyTorch ONNX export may require `onnxscript`.
- Opset version compatibility matters.
- Export success should be verified with `onnx.checker.check_model`.

Next step:

Run this ONNX model using ONNX Runtime.
