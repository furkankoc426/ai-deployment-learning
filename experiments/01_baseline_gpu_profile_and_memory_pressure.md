# Experiment 01 - Baseline GPU Profile and Memory Pressure

## Goal

Profile one model on GPU and answer practical deployment questions:

- how much memory does it use?
- how fast is it?
- what happens under a process memory cap?
- can two copies of the model run together?
- does FP16 reduce memory and latency?

## Model choice

Start with `torchvision.models.resnet50`.

Why ResNet50:

- small enough for Colab T4
- bigger than ResNet18
- easy to compare FP32 vs FP16
- good first profiling target before moving to ViT or LLMs

## Metrics to collect

For each scenario:

- avg latency ms
- p50 latency ms
- p95 latency ms
- throughput images/sec
- torch.cuda.max_memory_allocated
- torch.cuda.max_memory_reserved
- parameter memory MB
- nvidia-smi memory used before and after

## Scenarios

### Scenario 1 - FP32 baseline

Run ResNet50 with batch sizes:

```text
1, 8, 16, 32
```

Collect latency and memory.

### Scenario 2 - FP16 inference

Use autocast or convert model/input to half precision.

Expected:

- lower activation memory
- lower parameter memory if model weights are stored as FP16
- usually faster on Tensor Core capable GPUs

### Scenario 3 - process memory cap

Use:

```python
torch.cuda.set_per_process_memory_fraction(fraction)
```

Important expectation:

This is a cap on the PyTorch CUDA caching allocator. It does not magically optimize the model. If the workload needs more memory than allowed, it will fail with OOM. If it fits, latency may be similar.

Test fractions:

```text
0.90, 0.75, 0.60, 0.50, 0.40
```

### Scenario 4 - two model instances

Load two ResNet50 model instances on the same GPU.

Measure:

- memory after model 1 load
- memory after model 2 load
- sequential latency
- concurrent latency with CUDA streams

Expected:

- weight memory is close to additive
- activation memory depends on batch and overlap
- throughput may improve if GPU was underutilized
- latency may get worse due to contention

### Scenario 5 - hard partitioning discussion

Colab T4 does not support real MIG partitioning in the user session.

Real partition experiments require a GPU and cloud environment that supports one of these:

- NVIDIA MIG, commonly A100/H100 class GPUs
- NVIDIA MPS with active thread percentage control
- Kubernetes GPU sharing/time-slicing
- separate GPU instances from a cloud provider

## First Colab cell

```python
!nvidia-smi
```

## Baseline profiling cell

```python
import gc
import time
import statistics
import subprocess
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device('cuda')
BATCH_SIZES = [1, 8, 16, 32]
WARMUP = 20
ITERS = 100


def nvidia_smi_memory():
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'
    ]).decode().strip()
    used, total, util, power = [x.strip() for x in out.split(',')]
    return {
        'smi_mem_used_mb': float(used),
        'smi_mem_total_mb': float(total),
        'gpu_util_percent': float(util),
        'power_w': float(power),
    }


def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def param_memory_mb(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / 1024**2


def benchmark(model, batch_size, dtype=torch.float32):
    reset_cuda()
    model.eval().to(DEVICE)

    if dtype == torch.float16:
        model = model.half()

    x = torch.randn(batch_size, 3, 224, 224, device=DEVICE, dtype=dtype)

    before = nvidia_smi_memory()

    with torch.inference_mode():
        for _ in range(WARMUP):
            y = model(x)

    torch.cuda.synchronize()

    latencies = []
    with torch.inference_mode():
        for _ in range(ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            y = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1_000_000)

    after = nvidia_smi_memory()
    avg = statistics.mean(latencies)

    return {
        'batch_size': batch_size,
        'dtype': str(dtype).replace('torch.', ''),
        'avg_ms': avg,
        'p50_ms': statistics.median(latencies),
        'p95_ms': sorted(latencies)[int(len(latencies) * 0.95) - 1],
        'throughput_img_s': batch_size * 1000 / avg,
        'param_memory_mb': param_memory_mb(model),
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
        **{f'before_{k}': v for k, v in before.items()},
        **{f'after_{k}': v for k, v in after.items()},
    }


print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(nvidia_smi_memory())

rows = []
for dtype in [torch.float32, torch.float16]:
    for batch_size in BATCH_SIZES:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        rows.append(benchmark(model, batch_size, dtype=dtype))
        del model
        reset_cuda()

results = pd.DataFrame(rows)
print(results.round(3).to_string(index=False))
results.to_csv('experiment_01_baseline_profile.csv', index=False)
```

## Memory cap cell

Run this in a fresh runtime before loading the model.

```python
import torch

torch.cuda.set_per_process_memory_fraction(0.50, device=0)
print('Memory fraction set to 50 percent')
```

Then run the baseline profiling cell with smaller batch sizes first.

## Interpretation checklist

When reviewing results, answer:

1. What is the first batch size that fails under the cap?
2. If the model still fits under the cap, did latency change much?
3. How much memory did FP16 save?
4. Is reserved memory much higher than allocated memory?
5. Does max allocated memory scale roughly with batch size?
6. Can two model instances fit at the same time?

## Next experiment

Experiment 02 will test two-model concurrency using CUDA streams.
