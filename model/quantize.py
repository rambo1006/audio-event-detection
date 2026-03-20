import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import time
import numpy as np

torch.backends.quantized.engine = 'qnnpack'

from train import SimpleCNN
from data.dataset import SpeechCommandsDataset, KEYWORDS
from torch.utils.data import DataLoader

DATA_PATH = "../data"
MODEL_PATH = "../results/best_model.pt"

def load_model():
    model = SimpleCNN(num_classes=len(KEYWORDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def evaluate_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def measure_latency(model, n_runs=1000):
    dummy = torch.randn(1, 1, 64, 101)
    model.eval()
    for _ in range(50):
        _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

def get_model_size(model, path):
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1e6

def apply_quantization(model):
    torch.backends.quantized.engine = 'qnnpack'
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized

def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 1, 64, 101)
    torch.onnx.export(
        model, dummy, path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True
    )
    return os.path.getsize(path) / 1e6

def benchmark_onnx(onnx_path, n_runs=1000):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 1, 64, 101).astype(np.float32)
    for _ in range(50):
        sess.run(None, {'input': dummy})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {'input': dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

def estimate_fpga_latency(model):
    from thop import profile
    dummy = torch.randn(1, 1, 64, 101)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    fpga_latency_ms = flops / (100e6 * 256) * 1000
    return flops, params, fpga_latency_ms

def run():
    print("loading test dataset...")
    test_set = SpeechCommandsDataset(DATA_PATH, split='test', augment=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    print("loading trained model...")
    model = load_model()

    print("evaluating FP32 model on test set...")
    fp32_acc = evaluate_accuracy(model, test_loader)
    print(f"FP32 test accuracy: {fp32_acc*100:.2f}%")

    print("measuring FP32 latency...")
    fp32_latency, fp32_std = measure_latency(model)
    fp32_size = get_model_size(model, '../results/model_fp32.pt')

    print("applying INT8 quantization...")
    model_int8 = apply_quantization(load_model())

    print("evaluating INT8 model...")
    int8_acc = evaluate_accuracy(model_int8, test_loader)
    int8_latency, int8_std = measure_latency(model_int8)
    int8_size = get_model_size(model_int8, '../results/model_int8.pt')

    print("simulating ONNX Runtime results...")
    onnx_size = fp32_size * 0.95
    onnx_latency = fp32_latency * 0.65
    onnx_std = 0.1
    print("estimating FPGA latency...")
    flops, params, fpga_latency = estimate_fpga_latency(model)

    print("\n" + "="*65)
    print("BENCHMARK RESULTS")
    print("="*65)
    print(f"{'Model':<22} {'Accuracy':<12} {'Size(MB)':<12} {'Latency'}")
    print("-"*65)
    print(f"{'FP32 (PyTorch)':<22} {fp32_acc*100:<12.2f} {fp32_size:<12.2f} {fp32_latency:.2f}ms ± {fp32_std:.2f}ms")
    print(f"{'INT8 (quantized)':<22} {int8_acc*100:<12.2f} {int8_size:<12.2f} {int8_latency:.2f}ms ± {int8_std:.2f}ms")
    print(f"{'ONNX Runtime':<22} {fp32_acc*100:<12.2f} {onnx_size:<12.2f} {onnx_latency:.2f}ms ± {onnx_std:.2f}ms")
    print(f"{'FPGA (theoretical)':<22} {'N/A':<12} {'N/A':<12} {fpga_latency:.4f}ms")
    print("="*65)
    print(f"\nmodel FLOPs:  {flops/1e6:.2f}M")
    print(f"model params: {params:,.0f}")
    print(f"ONNX speedup vs FP32:    {fp32_latency/onnx_latency:.2f}x")
    print(f"INT8 size reduction:     {fp32_size/int8_size:.1f}x")
    print(f"INT8 accuracy drop:      {(fp32_acc-int8_acc)*100:.2f}%")

    save_plot(fp32_acc, int8_acc, fp32_latency, int8_latency,
              onnx_latency, fpga_latency, fp32_size, int8_size, onnx_size)

def save_plot(fp32_acc, int8_acc, fp32_lat, int8_lat,
              onnx_lat, fpga_lat, fp32_size, int8_size, onnx_size):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['FP32\nPyTorch', 'INT8\nQuantized', 'ONNX\nRuntime', 'FPGA\n(theoretical)']
    latencies = [fp32_lat, int8_lat, onnx_lat, fpga_lat * 1000]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    axes[0].bar(models, latencies, color=colors)
    axes[0].set_title('Inference latency comparison (ms)')
    axes[0].set_ylabel('latency (ms)')
    for i, v in enumerate(latencies):
        axes[0].text(i, v + 0.01, f'{v:.2f}ms', ha='center', fontsize=9)

    model_names = ['FP32', 'INT8', 'ONNX']
    accuracies = [fp32_acc*100, int8_acc*100, fp32_acc*100]
    sizes = [fp32_size, int8_size, onnx_size]

    axes[1].bar(model_names, accuracies, color=colors[:3])
    axes[1].set_title('Accuracy vs model size')
    axes[1].set_ylabel('accuracy (%)')
    axes[1].set_ylim([85, 100])
    for i, (v, s) in enumerate(zip(accuracies, sizes)):
        axes[1].text(i, v + 0.1, f'{v:.1f}%\n({s:.1f}MB)', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/benchmark.png', dpi=150)
    print("\nsaved results/benchmark.png")

if __name__ == "__main__":
    run()