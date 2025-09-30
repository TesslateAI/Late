# Llama 3 Training Examples

This directory contains example configurations for fine-tuning Meta's Llama 3 8B Instruct model using the Late framework.

## Files

### Training Configurations
- `sft_config.yml` - Full supervised fine-tuning (all parameters)
- `lora_config.yml` - Parameter-efficient fine-tuning with LoRA

### Sweep Configurations
- `llama3_lr_sweep.sweep` - Learning rate optimization
- `llama3_lora_sweep.sweep` - Comprehensive LoRA hyperparameter search
- `llama3_batch_sweep.sweep` - Batch size and gradient accumulation tuning

## Quick Start

### 1. Single Training Run
```bash
# Full fine-tuning (requires ~80GB VRAM)
late train examples/llama3/sft_config.yml

# LoRA fine-tuning (requires ~24GB VRAM)
late train examples/llama3/lora_config.yml
```

### 2. Hyperparameter Sweeps
```bash
# Quick learning rate search
late sweep examples/llama3/lora_config.yml --sweep-file examples/llama3/llama3_lr_sweep.sweep

# Comprehensive LoRA optimization
late sweep examples/llama3/lora_config.yml --sweep-file examples/llama3/llama3_lora_sweep.sweep

# Find optimal batch configuration for your GPU
late sweep examples/llama3/lora_config.yml --sweep-file examples/llama3/llama3_batch_sweep.sweep
```

### 3. Queue Multiple Experiments
```bash
# Create a queue for overnight runs
late queue create llama3_experiments.qml

# Add configurations
late queue add llama3_experiments.qml examples/llama3/lora_config.yml
late sweep examples/llama3/lora_config.yml --sweep-file examples/llama3/llama3_lr_sweep.sweep --add-to-queue llama3_experiments.qml

# Start the queue
late queue start llama3_experiments.qml
```

## Hardware Requirements

### Full Fine-Tuning (SFT)
- **Minimum VRAM**: 80GB (e.g., A100 80GB)
- **Recommended**: Multi-GPU setup with gradient checkpointing
- **Training Time**: ~12-24 hours per epoch on A100

### LoRA Fine-Tuning
- **Minimum VRAM**: 24GB (e.g., RTX 4090, A10)
- **Recommended**: 40GB+ for larger batch sizes
- **Training Time**: ~2-4 hours per epoch on A100

## Tips for Llama 3 Training

1. **Context Length**: Llama 3 supports up to 8K tokens. Start with 4K for LoRA to save memory.

2. **Learning Rates**:
   - SFT: 1e-5 to 2e-5 (conservative)
   - LoRA: 1e-4 to 3e-4 (can be higher)

3. **LoRA Rank**:
   - r=32: Minimal parameters, good for simple adaptations
   - r=128: Balanced performance/efficiency (recommended)
   - r=256: Maximum adaptation capacity

4. **Batch Size**: Use gradient accumulation to achieve effective batch size of 32-64

5. **Datasets**: 
   - Start with high-quality datasets like `HuggingFaceH4/no_robots`
   - For domain-specific tasks, mix general + domain data

## Monitoring Training

Enable W&B logging to track:
- Loss curves
- Learning rate schedules  
- GPU memory usage
- Gradient norms

```bash
# Set your W&B API key
late set wandb YOUR_API_KEY
```

## Common Issues

1. **Out of Memory**: 
   - Enable gradient checkpointing
   - Reduce batch size
   - Use LoRA instead of full fine-tuning
   - Reduce max_seq_length

2. **Slow Training**:
   - Enable torch_compile (10-20% speedup)
   - Use adamw_torch_fused optimizer
   - Ensure Flash Attention is installed

3. **Poor Convergence**:
   - Run learning rate sweep first
   - Check dataset quality
   - Ensure proper chat template formatting