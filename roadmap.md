# AI Deployment Roadmap

This roadmap is designed for learning AI deployment without a local GPU. All GPU exercises will use Colab, Kaggle Notebook, or another cloud GPU environment.

## Phase 1 - Inference basics

Goal: understand inference and how to measure it.

Topics:

- Training vs inference
- Latency
- Throughput
- Batch size
- Warmup
- CPU vs GPU inference
- Basic benchmarking mistakes

Output:

- Run a pretrained PyTorch model in Colab
- Measure latency correctly
- Save results in progress.md

## Phase 2 - PyTorch model inference

Goal: run pretrained models and understand input/output flow.

Topics:

- torchvision models
- eval mode
- torch.no_grad()
- input tensors
- image preprocessing
- prediction output

Output:

- ResNet18 inference script
- Latency numbers for CPU and GPU

## Phase 3 - ONNX export

Goal: convert a PyTorch model into ONNX.

Topics:

- ONNX format
- torch.onnx.export
- opset version
- static shape
- dynamic shape
- model validation

Output:

- resnet18.onnx
- notes about export settings

## Phase 4 - ONNX Runtime

Goal: run ONNX models outside PyTorch.

Topics:

- ONNX Runtime
- CPUExecutionProvider
- CUDAExecutionProvider
- input and output names
- PyTorch vs ONNX Runtime benchmark

Output:

- ONNX Runtime inference script
- benchmark table

## Phase 5 - CUDA and cuDNN basics

Goal: understand the stack without writing low-level CUDA first.

Topics:

- NVIDIA driver
- CUDA toolkit
- cuDNN
- GPU kernels
- why version mismatch happens
- how PyTorch uses CUDA

Output:

- short notes explaining the stack

## Phase 6 - TensorRT

Goal: optimize an ONNX model with TensorRT.

Topics:

- TensorRT engine
- trtexec
- FP32
- FP16
- unsupported operators
- TensorRT runtime

Output:

- TensorRT engine file
- benchmark comparison

## Phase 7 - Precision and quantization

Goal: understand speed vs accuracy trade-offs.

Topics:

- FP32
- FP16
- INT8
- calibration
- accuracy check

Output:

- FP32 vs FP16 benchmark
- INT8 notes

## Phase 8 - Benchmarking properly

Goal: measure inference in a reliable way.

Topics:

- warmup
- torch.cuda.synchronize()
- average latency
- p50 and p95 latency
- throughput
- memory usage

Output:

- reusable benchmark helper

## Phase 9 - FastAPI model serving

Goal: serve a model through an API.

Topics:

- FastAPI
- /predict endpoint
- loading model once at startup
- image upload
- JSON response
- latency logging

Output:

- local FastAPI app
- simple API test

## Phase 10 - Docker basics

Goal: understand deployment packaging.

Topics:

- Dockerfile
- image
- container
- dependencies
- environment reproducibility
- NVIDIA Container Toolkit concept

Output:

- Dockerfile for CPU inference
- notes for GPU containers

## Phase 11 - Triton Inference Server

Goal: understand production-grade model serving.

Topics:

- model repository
- config.pbtxt
- ONNX deployment
- TensorRT deployment
- dynamic batching
- model versioning

Output:

- Triton model repo example
- notes about HTTP/gRPC inference
