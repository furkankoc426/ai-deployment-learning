# 2026-07-09 ResNet18 Benchmark

Environment:

```text
PyTorch version: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
CUDA version: 12.8
```

Model: torchvision ResNet18 pretrained
Input shape: batch x 3 x 224 x 224
Warmup iterations: 10
Measurement iterations: 50

## Results

| Batch size | Device | Avg latency ms | P50 latency ms | P95 latency ms | Throughput images/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | CPU | 59.506 | 57.980 | 72.207 | 16.805 |
| 1 | GPU | 3.906 | 3.879 | 3.897 | 256.017 |
| 8 | CPU | 443.711 | 407.358 | 599.045 | 18.030 |
| 8 | GPU | 10.132 | 9.274 | 13.134 | 789.566 |
| 16 | CPU | 927.116 | 872.697 | 1232.204 | 17.258 |
| 16 | GPU | 14.636 | 14.591 | 14.971 | 1093.214 |

## Interpretation

GPU is much faster than CPU for this model and runtime.

Batch size 1:

- CPU avg latency: 59.506 ms
- GPU avg latency: 3.906 ms
- GPU is about 15.2x faster by average latency.

Batch size 16:

- CPU throughput: 17.258 images/sec
- GPU throughput: 1093.214 images/sec
- GPU is about 63.3x higher throughput.

## Key learning

GPU advantage becomes very clear when batch size increases. Latency increases with batch size, but throughput also increases strongly on GPU.

Next topic: export the PyTorch model to ONNX.
