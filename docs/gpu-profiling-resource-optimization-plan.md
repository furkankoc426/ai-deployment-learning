# GPU Profiling and Resource Optimization Plan

This plan changes the direction of the repo from basic ONNX learning to practical deployment profiling.

## Scenario we want to study

Example target scenario:

- GPU memory: 6 GB
- Model without optimization: 3.5 GB peak memory
- Latency without optimization: 20 ms

Questions:

- If we force the model to stay near 2.5 GB, does it run slower or fail?
- Which optimizations actually reduce memory?
- Can two models run at the same time on one GPU?
- When does concurrency improve throughput?
- When does concurrency hurt latency?
- What is a real hardware partition, and what is only a software-level cap?

## Important distinction

A memory limit is not the same thing as optimization.

If a PyTorch process is capped below the memory it needs, the most likely result is OOM. The model will not automatically become more efficient.

To reduce memory and still run, one of these must change:

- batch size
- precision: FP32 to FP16 or BF16
- model architecture
- activation footprint
- TensorRT tactics and workspace
- quantization
- CPU offload
- KV cache size for LLMs
- number of concurrent model replicas

## Profiling layers

### Layer 1 - Framework-level metrics

Use PyTorch CUDA memory APIs:

- torch.cuda.memory_allocated
- torch.cuda.memory_reserved
- torch.cuda.max_memory_allocated
- torch.cuda.max_memory_reserved
- torch.cuda.reset_peak_memory_stats

These tell us what the PyTorch allocator sees.

### Layer 2 - Device-level metrics

Use nvidia-smi:

- memory.used
- utilization.gpu
- utilization.memory
- power.draw
- temperature.gpu

These tell us what the GPU driver sees.

### Layer 3 - Operator-level metrics

Use torch.profiler for:

- operator time
- CUDA kernel time
- memory usage per operator
- CPU overhead

This helps answer which layers dominate time or memory.

### Layer 4 - Engine-level metrics

For TensorRT:

- engine build success or failure
- workspace limit
- selected tactics
- FP32 vs FP16
- serialized engine size
- runtime latency
- runtime memory usage

TensorRT workspace limit can change tactic selection. Lower workspace may reduce temporary memory but can also make TensorRT choose slower kernels or fail to build an engine.

### Layer 5 - Partitioning and multi-tenancy

Three categories:

1. Soft cap
   - PyTorch memory fraction
   - process-level limit
   - does not give true hardware isolation

2. Scheduling / sharing
   - CUDA streams
   - multiple processes
   - NVIDIA MPS where available
   - good for utilization, but interference is possible

3. Hardware partition
   - NVIDIA MIG on supported data center GPUs
   - isolated GPU instances with dedicated compute and memory slices
   - not available on Colab T4

## Experiment order

### Experiment 01 - Baseline model profile

Model: ResNet50 or ViT-B/16

Measure:

- model parameter memory
- peak allocated memory
- peak reserved memory
- latency avg, p50, p95
- throughput
- GPU utilization snapshot

### Experiment 02 - Batch-size scaling

Run batch sizes:

- 1
- 2
- 4
- 8
- 16
- 32 if memory allows

Goal:

Find memory and latency scaling behavior.

### Experiment 03 - Memory cap test

Use PyTorch memory fraction cap.

Run with caps:

- no cap
- 90 percent of baseline peak
- 75 percent of baseline peak
- 60 percent of baseline peak

Expected result:

- if model fits, latency may be similar
- if the cap is too low, OOM
- memory cap alone does not compress the model

### Experiment 04 - Memory reduction knobs

Test:

- smaller batch size
- FP16 autocast
- channels_last
- torch.compile if supported

Compare:

- memory reduction
- latency change
- numerical output difference

### Experiment 05 - Two model concurrency

Run two copies of the same model:

- sequential execution
- concurrent CUDA streams in one process
- two separate processes if environment allows

Measure:

- total throughput
- latency per model
- peak memory
- whether both fit at the same time

### Experiment 06 - TensorRT workspace and precision

Build engines with:

- FP32 default workspace
- FP16 default workspace
- smaller workspace limits

Measure:

- build success
- engine size
- runtime memory
- latency

### Experiment 07 - Real partitioning notes

If using a MIG-capable GPU later, test:

- full GPU
- smaller MIG slice
- two MIG slices running two models

Colab T4 cannot do this, so this phase is conceptual unless we get A100/H100/L40S type access with MIG enabled.

## First practical target

Start with Experiment 01 and 03 together:

1. Choose ResNet50.
2. Measure baseline memory and latency on T4.
3. Apply PyTorch memory caps.
4. See whether it runs, slows down, or OOMs.
5. Record all results in a markdown log.
