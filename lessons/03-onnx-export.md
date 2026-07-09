# Lesson 03 - Export PyTorch ResNet18 to ONNX

## Goal

Convert a PyTorch ResNet18 model into ONNX format.

Flow:

```text
PyTorch ResNet18 -> resnet18.onnx
```

## Why ONNX matters

ONNX is an open model format. It helps move models between frameworks and runtimes.

In deployment, ONNX is useful because the same exported model can be used by tools such as:

- ONNX Runtime
- TensorRT
- Triton Inference Server

## Important concepts

### 1. Dummy input

ONNX export needs an example input tensor. This tells the exporter what input shape the model expects.

For ResNet18 image classification:

```python
x = torch.randn(1, 3, 224, 224)
```

Meaning:

```text
batch size = 1
channels = 3
height = 224
width = 224
```

### 2. Static shape

A static-shape ONNX model expects a fixed input shape.

Example:

```text
1 x 3 x 224 x 224
```

This is simple and good for the first export.

### 3. Dynamic shape

A dynamic-shape ONNX model can accept different batch sizes.

Example:

```text
batch x 3 x 224 x 224
```

We will start with static shape first, then later learn dynamic shape.

### 4. Opset version

Opset is the ONNX operator version used during export.

For this lesson we will use:

```text
opset_version = 17
```

### 5. Input and output names

Naming inputs and outputs makes later ONNX Runtime code easier.

Example:

```text
input name: input
output name: output
```

## Colab exercise

Install ONNX:

```python
!pip install onnx
```

Export model:

```python
import torch
import onnx
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

x = torch.randn(1, 3, 224, 224)
onnx_path = "resnet18_static.onnx"

torch.onnx.export(
    model,
    x,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("ONNX export successful:", onnx_path)
print("Input name:", onnx_model.graph.input[0].name)
print("Output name:", onnx_model.graph.output[0].name)
print("Number of nodes:", len(onnx_model.graph.node))
```

## Expected result

You should see something like:

```text
ONNX export successful: resnet18_static.onnx
Input name: input
Output name: output
Number of nodes: ...
```

## Homework

Run the Colab code and send back:

- whether export succeeded
- input name
- output name
- number of nodes
- file size if possible

Next lesson: run this ONNX model with ONNX Runtime.
