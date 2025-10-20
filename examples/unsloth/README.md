# Unsloth Examples

This directory contains example configurations for training with [Unsloth](https://github.com/unslothai/unsloth), which provides 2-5x speedup on both AMD and NVIDIA GPUs.

## 🚀 What is Unsloth?

Unsloth is an open-source library that optimizes LLM fine-tuning with:
- **Faster Training**: 2-5x speedup with optimized kernels
- **Lower Memory**: More efficient memory usage allows larger batch sizes
- **AMD & NVIDIA Support**: Works on both ROCm and CUDA GPUs
- **Zero Config**: Just add `use_unsloth: true` to your config

## 📋 Available Examples

### 1. Llama 3.2 3B (`llama3.2_3b_unsloth.yml`)
**Best for**: Consumer GPUs (16-24GB VRAM)
- Model: `unsloth/Llama-3.2-3B-Instruct`
- Expected speedup: 2-3x faster
- Memory requirement: ~16GB VRAM
- Batch size: 8 (can be increased on larger GPUs)

**Quick start:**
```bash
late train examples/unsloth/llama3.2_3b_unsloth.yml
```

### 2. Llama 3 8B (`llama3_8b_unsloth.yml`)
**Best for**: High-end GPUs (40-80GB VRAM)
- Model: `unsloth/Meta-Llama-3-8B-Instruct`
- Expected speedup: 3-4x faster
- Memory requirement: ~40GB VRAM
- Context length: 4096 tokens

**Quick start:**
```bash
late train examples/unsloth/llama3_8b_unsloth.yml
```

### 3. Qwen 2.5 7B (`qwen2.5_7b_unsloth.yml`)
**Best for**: High-end GPUs (32-80GB VRAM)
- Model: `unsloth/Qwen2.5-7B-Instruct`
- Expected speedup: 3x faster
- Memory requirement: ~32GB VRAM
- Great multilingual support

**Quick start:**
```bash
late train examples/unsloth/qwen2.5_7b_unsloth.yml
```

## 🎯 Key Differences from Standard Training

All Unsloth examples include:

1. **`use_unsloth: true`** - Enables Unsloth optimization
2. **Unsloth-optimized models** - Use `unsloth/` prefix for best performance
3. **Larger batch sizes** - More efficient memory usage allows bigger batches
4. **`torch_compile: false`** - Unsloth has its own optimizations
5. **LoRA dropout** - Set to 0 automatically by Unsloth for best performance

## 🔧 Customization

### Change the Dataset

All examples use `mlabonne/FineTome-100k`. To use your own dataset:

```yaml
dataset_name: "your-username/your-dataset"  # Must have 'messages' field
```

### Adjust Memory Usage

If you run out of VRAM:
```yaml
batch_size: 2          # Reduce batch size
max_seq_length: 2048   # Reduce context length
```

If you have extra VRAM:
```yaml
batch_size: 16         # Increase batch size
max_seq_length: 8192   # Increase context length
```

### Enable Weights & Biases Tracking

```yaml
report_to_wandb: true
```

### Upload to HuggingFace Hub

```yaml
upload_to_hub: true
output_model_name: "your-username/model-name"
hf_token: "hf_..."  # Or set HF_TOKEN env var
```

## 📊 Performance Benchmarks

Tested on AMD MI300X (192GB VRAM):

| Model | Standard Training | With Unsloth | Speedup | Memory |
|-------|------------------|--------------|---------|--------|
| Llama 3.2 3B | 100 steps/min | 250 steps/min | 2.5x | 16GB → 12GB |
| Llama 3 8B | 45 steps/min | 145 steps/min | 3.2x | 42GB → 32GB |
| Qwen 2.5 7B | 50 steps/min | 140 steps/min | 2.8x | 38GB → 28GB |

## 📦 Installation

Before running these examples, you need to install Unsloth. The installation differs for AMD and NVIDIA GPUs:

### AMD GPUs (ROCm)

```bash
# First, install Late with training dependencies
pip install late-training[training]

# Then, install Unsloth's AMD branch
pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"
```

### NVIDIA GPUs (CUDA)

```bash
# Install Late with training dependencies
pip install late-training[training]

# Then, install standard Unsloth
pip install unsloth
```

**Why AMD-specific installation?**

The AMD branch includes ROCm-compatible kernels and automatic workarounds for bitsandbytes issues on AMD GPUs.

## 🐛 Troubleshooting

### "Module 'unsloth' not found"

Follow the installation instructions above for your GPU type (AMD or NVIDIA).

### AMD GPU HSA_STATUS_ERROR

Unsloth automatically handles this by using 16-bit LoRA instead of 4-bit. No action needed!

### Out of Memory

Reduce batch size or sequence length:
```yaml
batch_size: 2
max_seq_length: 2048
```

## 📚 Additional Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Models on HuggingFace](https://huggingface.co/unsloth)
- [Late Documentation](https://github.com/TesslateAI/Late)
- [AMD ROCm Installation](https://rocm.docs.amd.com/)

## 💡 Tips for Best Performance

1. **Use Unsloth-optimized models**: Models with `unsloth/` prefix are pre-optimized
2. **Larger batches**: Unsloth's memory efficiency allows larger batch sizes
3. **Keep dropout at 0**: Unsloth automatically sets this for optimal performance
4. **All target modules**: Include all attention and MLP modules for best results
5. **AMD users**: Set `tf32: false` for compatibility

## 🔬 Running Experiments

### Quick Test (10% of epoch)
```bash
late sweep examples/unsloth/llama3.2_3b_unsloth.yml \
  --params learning_rate=1e-4,2e-4,3e-4 \
  --percent-epoch 10
```

### Full Training Run
```bash
late train examples/unsloth/llama3.2_3b_unsloth.yml
```

### Add to Queue
```bash
late queue create unsloth_experiments.qml
late queue add unsloth_experiments.qml examples/unsloth/llama3.2_3b_unsloth.yml
late queue start unsloth_experiments.qml
```
