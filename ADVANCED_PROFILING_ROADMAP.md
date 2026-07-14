# Advanced AI Deployment Roadmap: GPU Profiling and Resource Partitioning

This document replaces the basic step-by-step ONNX learning flow with a more practical deployment-oriented plan.

Main goal:

Profile a model on GPU, understand memory and latency behavior, then test what happens under memory pressure, reduced optimization workspace, lower precision, and concurrent model execution.

## Target questions

1. How much GPU memory does a model use?
2. How much of that is model weights, activations, workspace, allocator reserve, and temporary buffers?
3. How long does inference take at different batch sizes?
4. What happens if the process is allowed to use less GPU memory?
5. Can two models run on the same GPU at the same time?
6. When does concurrency improve throughput, and when does it hurt latency?
7. What is the difference between soft limits, runtime limits, and real hardware partitioning?
8. Which optimizations reduce memory, and what latency cost do they introduce?

## New learning flow

### Phase A - Baseline GPU profiling

Pick one model and collect baseline metrics:

- latency: avg, p50, p95
- throughput
- peak allocated GPU memory
- peak reserved GPU memory
- parameter memory
- activation and temporary memory estimate
- GPU utilization via nvidia-smi
- optional operator-level profile via torch.profiler

Initial model choices:

- resnet50 for simple vision profiling
- efficientnet_b0 for lighter model
- vit_b_16 for memory-heavy vision transformer
- later: small LLM for KV cache and serving behavior

### Phase B - Memory pressure experiments

Run the same model under different memory caps.

Example scenarios:

- no limit
- 90 percent of required memory
- 75 percent of required memory
- 60 percent of required memory
- 50 percent of required memory

Important expectation:

A memory cap alone does not automatically make a model use less memory. In PyTorch it usually either fits or fails with OOM. To reduce memory and still run, something must change:

- smaller batch size
- lower precision: FP32 -> FP16/BF16
- model quantization
- TensorRT tactic/workspace restriction
- CPU offload
- recomputation/checkpointing, mostly for training
- smaller model architecture
- shorter sequence length for LLMs

### Phase C - Optimization knobs

Test knobs one by one and measure memory and latency.

Knobs:

- batch size
- FP32 vs FP16/autocast
- channels_last for vision models
- torch.compile where supported
- ONNX Runtime CUDA provider
- TensorRT engine with FP16
- TensorRT workspace limit
- model quantization when possible

### Phase D - Concurrent model execution

Test whether two models can run at the same time on one GPU.

Scenarios:

- one model only
- two model instances loaded, run sequentially
- two model instances loaded, run with CUDA streams
- two different models loaded, run sequentially
- two different models loaded, run concurrently

Metrics:

- total GPU memory
- per-request latency
- total throughput
- p95 latency
- failure mode when memory is insufficient

### Phase E - Partitioning concepts

Understand what can and cannot be done in Colab.

Colab T4 limitations:

- no MIG hardware partitioning
- no admin-level GPU partition control
- limited nvidia-smi control
- cannot reliably configure MPS or Kubernetes GPU sharing

Real deployment options:

- MIG on supported data-center GPUs such as A100/H100 class devices
- MPS for process-level sharing on supported setups
- Kubernetes GPU time-slicing or MIG device plugin
- TensorRT memory pool/workspace limits
- Triton model instance count and dynamic batching
- vLLM/TensorRT-LLM memory controls for LLM serving

## Experiment sequence

1. Baseline ResNet50 PyTorch GPU profiling.
2. Repeat with batch sizes 1, 8, 16, 32 if memory allows.
3. Profile FP32 vs FP16.
4. Apply PyTorch per-process memory fraction and observe fit vs OOM behavior.
5. Run two model instances sequentially and concurrently.
6. Export to ONNX and run ONNX Runtime CUDA if available.
7. Build TensorRT engine and test workspace limits if TensorRT is available in the environment.
8. Move to a larger model such as ViT or a small transformer.
9. Document when memory, compute, or bandwidth becomes the bottleneck.

## Key mental model

There are three different things that are often confused:

1. Memory cap

A hard or soft limit on what the process/runtime may allocate.

2. Memory optimization

Changing the model or runtime so it actually needs less memory.

3. GPU partitioning

Dividing physical GPU compute and memory resources between workloads.

A memory cap alone is not an optimization. It only exposes whether the current workload fits.

## Next experiment

Run experiment 01:

experiments/01_pytorch_gpu_profiling_memory_pressure.md
