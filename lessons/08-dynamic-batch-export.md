# Lesson 08 - Export ONNX with a dynamic batch dimension

## Goal

Export one ResNet18 ONNX model that accepts multiple batch sizes:

```text
batch 1  -> output (1, 1000)
batch 8  -> output (8, 1000)
batch 16 -> output (16, 1000)
```

Lesson 07 measured batch-size-1 latency. Before benchmarking larger batches, the ONNX model must accept more than the example batch used during export.

## Static vs dynamic batch

A normal export records the example input shape:

```text
[1, 3, 224, 224]
```

This makes axis 0 static. A batch of 8 may then fail because the model expects exactly 1 image.

A dynamic batch model instead has a symbolic first dimension:

```text
[batch, 3, 224, 224]
```

The channel, height, and width remain fixed.

Dynamic shape support does not automatically combine independent requests into a batch. It only allows the model to accept different batch sizes. Request batching is a serving-layer topic for a later lesson.

## PyTorch 2.11 API

The modern ONNX exporter uses `dynamo=True` and `dynamic_shapes`.

We describe the batch dimension with:

```python
batch_dim = torch.export.Dim("batch", min=1, max=16)
```

For torchvision ResNet, the forward method is conceptually:

```python
def forward(self, x):
    ...
```

Therefore the `dynamic_shapes` dictionary uses the argument name `x`:

```python
dynamic_shapes={"x": {0: batch_dim}}
```

Do not replace `x` with the ONNX name `input`. `input_names=["input"]` renames the exported ONNX input, while `dynamic_shapes` describes the original Python forward arguments.

## Colab preparation

Continue in the clean GPU environment from Lesson 07. In a fresh runtime, use:

```python
!pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1
!pip install -q onnx onnxscript onnxruntime-gpu
```

If ONNX Runtime was imported before installation, restart the Colab session once.

## Colab exercise

```python
import os
import numpy as np
import torch
import onnx

# Import torch first so Colab CUDA/cuDNN libraries are available to ORT.
import onnxruntime as ort

from torchvision.models import resnet18, ResNet18_Weights

ONNX_PATH = "resnet18_dynamic_batch.onnx"
BATCH_SIZES = [1, 8, 16]


torch.manual_seed(42)
np.random.seed(42)

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
).eval()

# The example input is still batch size 1.
example_input = torch.randn(
    1, 3, 224, 224,
    dtype=torch.float32,
)

# Axis 0 may vary from 1 to 16 at runtime.
batch_dim = torch.export.Dim(
    "batch",
    min=1,
    max=16,
)

# ResNet.forward uses the Python argument name `x`.
dynamic_shapes = {
    "x": {0: batch_dim}
}

torch.onnx.export(
    model,
    (example_input,),
    ONNX_PATH,
    opset_version=18,
    dynamo=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_shapes=dynamic_shapes,
    external_data=False,
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

available_providers = ort.get_available_providers()
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers
    else ["CPUExecutionProvider"]
)

session = ort.InferenceSession(
    ONNX_PATH,
    providers=providers,
)

input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]

print("PyTorch version:", torch.__version__)
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", available_providers)
print("Session providers:", session.get_providers())
print("Model file size MB:", round(os.path.getsize(ONNX_PATH) / 1024**2, 2))
print("Input name:", input_meta.name)
print("Input shape:", input_meta.shape)
print("Output name:", output_meta.name)
print("Output shape:", output_meta.shape)

# The first dimension should be symbolic rather than integer 1.
assert not isinstance(input_meta.shape[0], int), input_meta.shape
assert input_meta.shape[1:] == [3, 224, 224], input_meta.shape

for batch_size in BATCH_SIZES:
    x = np.random.randn(
        batch_size, 3, 224, 224
    ).astype(np.float32)

    output = session.run(
        [output_meta.name],
        {input_meta.name: x},
    )[0]

    print(
        f"batch={batch_size}: "
        f"input={x.shape}, output={output.shape}"
    )

    assert output.shape == (batch_size, 1000)

print("PASS: one ONNX model accepted batch sizes 1, 8, and 16.")
```

## Expected output

The symbolic dimension name may differ, but the shapes should look similar to:

```text
Input shape: ['batch', 3, 224, 224]
Output shape: ['batch', 1000]
batch=1: input=(1, 3, 224, 224), output=(1, 1000)
batch=8: input=(8, 3, 224, 224), output=(8, 1000)
batch=16: input=(16, 3, 224, 224), output=(16, 1000)
PASS: one ONNX model accepted batch sizes 1, 8, and 16.
```

The exporter may choose a generated symbolic name instead of exactly `batch`. That is fine; the important result is that the first dimension is not fixed to integer `1`.

## Common failure

If export reports that `dynamic_shapes` does not match the input arguments, confirm this exact mapping:

```python
dynamic_shapes={"x": {0: batch_dim}}
```

Here `x` is the Python `forward` argument. The ONNX input is named `input` only after export.

## Save your work

Save the executed notebook over:

```text
notebooks/ai_deployment_learning.ipynb
```

Suggested commit message:

```text
lesson 08 export dynamic batch onnx model
```

## Next lesson

Benchmark latency and throughput for batch sizes 1, 8, and 16 using the same dynamic ONNX model.
