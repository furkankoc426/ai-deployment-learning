# Lesson 01 - Inference, Latency, Throughput

## Goal

Understand the basic language of AI deployment:

- inference
- latency
- throughput
- batch size
- warmup
- CPU vs GPU timing

## 1. Training vs inference

Training updates model weights.

Inference uses already trained weights to produce an output.

Training flow:

```text
input -> model -> loss -> backward -> weight update
```

Inference flow:

```text
input -> model -> output
```

In deployment, the main focus is inference.

## 2. Latency

Latency is the time needed to process one request.

Example:

```text
1 image -> model -> prediction in 12 ms
latency = 12 ms
```

Low latency is important for real-time systems.

## 3. Throughput

Throughput is how many items the system can process per second.

Example:

```text
300 images processed in 1 second
throughput = 300 images/second
```

Latency and throughput are related but not the same.

## 4. Batch size

Batch size means how many inputs are processed together.

Example:

```text
batch size = 1   -> one image at a time
batch size = 16  -> sixteen images together
```

A larger batch can improve throughput, but it can also increase latency for one request.

## 5. Warmup

The first few GPU inference calls can be slower because the framework prepares CUDA context, memory, and kernels.

So before measuring performance, run warmup iterations.

Example:

```python
for _ in range(10):
    _ = model(x)
```

## 6. GPU timing problem

CUDA operations are often asynchronous. This means Python can continue before the GPU work is actually finished.

Bad timing example:

```python
start = time.time()
_ = model(x)
end = time.time()
```

Better timing on GPU:

```python
torch.cuda.synchronize()
start = time.time()
_ = model(x)
torch.cuda.synchronize()
end = time.time()
```

## 7. First homework

Open Google Colab and enable GPU:

Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU

Run:

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

Send the output before moving to the next step.

## 8. What to record

Add the output to progress.md:

- PyTorch version
- CUDA available true/false
- GPU name
- CUDA version
