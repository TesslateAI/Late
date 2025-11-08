<div align="center">
  <img src="assets/late-logo.png" alt="Late Logo" width="200">

  # Late

  **A powerful toolkit for streamlining and scheduling ML training workflows on ROCm**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyPI version](https://badge.fury.io/py/late-training.svg)](https://badge.fury.io/py/late-training)
  [![Downloads](https://pepy.tech/badge/late-training)](https://pepy.tech/project/late-training)
  [![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://www.amd.com/en/products/software/rocm.html)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![GitHub stars](https://img.shields.io/github/stars/TesslateAI/Late.svg?style=social&label=Star)](https://github.com/TesslateAI/Late)
  [![GitHub forks](https://img.shields.io/github/forks/TesslateAI/Late.svg?style=social&label=Fork)](https://github.com/TesslateAI/Late/fork)
  [![GitHub issues](https://img.shields.io/github/issues/TesslateAI/Late.svg)](https://github.com/TesslateAI/Late/issues)
  [![GitHub pull requests](https://img.shields.io/github/issues-pr/TesslateAI/Late.svg)](https://github.com/TesslateAI/Late/pulls)
  [![Contributors](https://img.shields.io/github/contributors/TesslateAI/Late.svg)](https://github.com/TesslateAI/Late/graphs/contributors)
  [![Last Commit](https://img.shields.io/github/last-commit/TesslateAI/Late.svg)](https://github.com/TesslateAI/Late/commits/main)

  [Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Contributing](#-contributing)
</div>

---

<div align="center">
  <img src="assets/late-banner.png" alt="Late Training Dashboard" width="800">
</div>

## 📖 Overview

`Late` is a comprehensive Python toolkit that provides a unified interface for managing the entire lifecycle of training large language models on AMD GPUs. It combines a powerful **Command-Line Interface (CLI)**, a **batch job runner**, and a user-friendly **web dashboard** to simplify environment setup, job scheduling, and execution.

### Why Late?

- 🚀 **ROCm Optimized**: Built specifically for AMD GPUs with optimized defaults
- 🔧 **Zero Config**: Smart defaults that just work out of the box
- 📊 **Hyperparameter Sweeps**: Built-in sweep engine with automatic reporting
- 🔄 **Queue Management**: Batch processing with pause/resume capabilities
- 📈 **Real-time Monitoring**: Web dashboard for tracking training progress
- 🧪 **Reproducible**: Every run is tracked with configs and outputs saved

### 🎯 Feature Comparison

| Feature | Late | Other Tools |
|---------|------|-------------|
| ROCm Native Support | ✅ First-class | ⚠️ Limited |
| Hyperparameter Sweeps | ✅ Built-in with reports | ❌ External tools needed |
| Queue Management | ✅ Pause/Resume/Priority | ❌ Basic or none |
| Memory Management | ✅ Automatic VRAM clearing | ❌ Manual |
| Training Configs | ✅ Simple YAML | ⚠️ Complex scripts |
| Web Dashboard | ✅ Included | ❌ Separate setup |
| Reproducibility | ✅ Auto-saved runs | ⚠️ Manual tracking |

---

## ✨ Key Features

-   **Automated Environment Patching**: A single command (`late patch`) installs and configures Flash Attention for ROCm, targeting either your global environment or a specific Python virtual environment. By default uses pre-built wheels, with optional source building.
-   **Declarative Training Jobs**: Define all aspects of your training runs—from model choice to hyperparameters—in simple, readable YAML configuration files.
-   **Direct Training**: Run training jobs immediately with `late train config.yml` - no queue required for single experiments.
-   **Batch Queue Management**: Group multiple training configs into `.qml` queue files to run them sequentially. Perfect for running a series of experiments overnight.
-   **Versatile CLI**: Manage every aspect of your workflow from the terminal, including patching, training, and queue management.
-   **Web Dashboard**: Launch a web server (`late serve`) to visually create, manage, and monitor your training queues from any browser.
-   **Built for ROCm**: Optimized with defaults like `adamw_torch_fused` and `bfloat16` to get the best performance out of AMD hardware.

### 🆕 What's New

-   **⚡ Unsloth Integration**: Optional 2-5x speedup on both AMD and NVIDIA GPUs with optimized kernels
-   **🎯 Configurable Loss Masking**: Choose between "full" (default, simpler) or "assistant_only" loss masking strategies
-   **🔔 Push Notifications**: Get ntfy.sh notifications on checkpoint saves, training completion, and errors
-   **🔀 LoRA Merging**: Built-in command to merge LoRA adapters with base models and upload to Hub
-   **✅ Comprehensive Testing**: 80%+ test coverage with pytest, ready for CI/CD
-   **📚 Better Documentation**: Complete testing guide and example configurations

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **ROCm**: 6.0+ (for GPU training)
- **Git**: For installation and version control
- **OS**: Linux (Ubuntu/RHEL recommended)

### Install from PyPI (Recommended)

```bash
# Install the package
pip install late-training

# Verify installation
late version
```

### Install from GitHub (Development)

```bash
# Clone the repository
git clone https://github.com/TesslateAI/Late.git
cd Late

# Install in development mode
pip install -e .

# Verify installation
late version
```

### Install Dependencies

```bash
# Install core training dependencies
pip install late-training[training]

# For Unsloth support on AMD GPUs (optional but recommended for 2-5x speedup)
pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"

# For Unsloth support on NVIDIA GPUs (optional)
pip install unsloth
```

## 🚀 Quick Start

### 1️⃣ Setup Environment (First Time Only)

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Export your huggingface token
export HF_TOKEN=<your huggingface token>

# Clone the test dataset
git clone https://huggingface.co/datasets/yahma/alpaca-cleaned

# Patch environment for ROCm (installs Flash Attention, etc.)
late patch --arch gfx942  # Replace with your GPU architecture
```

### 2️⃣ Run Your First Training

```bash
# Create a simple LoRA training config
cat > quick_lora.yml << EOF
base_model: "meta-llama/Llama-3.2-3B-Instruct"
dataset_name: "alpaca-cleaned"
output_model_name: "my-first-lora"
output_dir: "./outputs/quick-test/"
training_type: "lora"
max_seq_length: 2048
batch_size: 4
gradient_accumulation: 4
epochs: 1
learning_rate: 2e-4
lora:
  r: 32
  lora_alpha: 64
EOF

# Start training immediately
late train quick_lora.yml
```

### 3️⃣ Run a Hyperparameter Sweep

```bash
# Find the best learning rate in minutes
late sweep quick_lora.yml \
  --params learning_rate=1e-4,2e-4,3e-4 \
  --percent-epoch 10  # Only train 10% of epoch for quick evaluation
```

### 4️⃣ Launch the Web Dashboard

```bash
# Start the web UI
late serve

# Open http://localhost:8080 in your browser
```


## 🎯 Core Concepts

Late is built around two simple file types: YAML for defining *what* to run, and QML for defining the *order* to run it in.

### 1. Training Configuration (`.yml`)

A YAML file defines a single training job. It contains all the necessary parameters, such as the model, dataset, hyperparameters, and training type (e.g., SFT or LoRA).

**Example `sft_config.yml`:**
```yaml
# Model and Dataset
base_model: "Qwen/Qwen2-1.5B-Instruct"
dataset_name: "Tesslate/UIGENT"
output_model_name: "your-hf-username/UIGENT-1.5B-SFT"

# Paths
output_dir: "/scratch/outputs/sft/"
cache_dir: "./model_cache"

# Training Type ('sft' or 'lora')
training_type: "sft"

# Hyperparameters
max_seq_length: 4096
batch_size: 2
gradient_accumulation: 8
epochs: 1
learning_rate: 2.0e-5

# Control Flags
report_to_wandb: true
upload_to_hub: false
```

### 2. Training Queues (`.qml`)

A QML file is a plain text file that lists the absolute paths to your YAML configuration files. The `late` runner will execute these jobs one by one, in the order they appear.

**Example `experiment_queue.qml`:**
```qml
/home/user/projects/late/configs/sft_run_1.yml
/home/user/projects/late/configs/lora_run_alpha.yml
/home/user/projects/late/configs/sft_run_2.yml
```

## 💻 Command-Line Interface (CLI) Usage

The `late` CLI is the primary way to interact with the library.

<div align="center">
  <img src="assets/cli.png" alt="Late CLI Commands" width="700">
</div>

### Environment Patching (`late patch`)

Prepares a Python environment for ROCm training. It automatically installs PyTorch ROCm, Flash Attention and other core ML libraries. 

**IMPORTANT**: PyTorch installation is now OPTIONAL by default. You must explicitly use `--install-pytorch` to install PyTorch ROCm versions.

**Usage:**
```bash
# Patch current environment WITHOUT PyTorch (default)
late patch --arch gfx942

# Install PyTorch stable ROCm version
late patch --arch gfx942 --install-pytorch stable

# Install PyTorch nightly ROCm 6.4 version
late patch --arch gfx942 --install-pytorch nightly

# Install PyTorch from specific wheel URL
late patch --arch gfx942 --install-pytorch https://example.com/torch-2.0-rocm.whl

# Build Flash Attention from source
late patch --arch gfx942 --from-source

# Create a venv and patch it specifically (recommended)
python3 -m venv my_env
late patch --venv ./my_env

# Multiple GPU architectures
late patch --arch "gfx942;gfx90a"

# Skip MIOpen kernel installation
late patch --no-kernels

# Full example: venv + PyTorch nightly + Flash from source
late patch --venv ./my_env --arch gfx942 --install-pytorch nightly --from-source
```

### Running a Single Training Job (`late train`)

Run a training job directly from a YAML config file without using queues. Perfect for quick experiments or single runs.

<div align="center">
  <img src="assets/training_cli.png" alt="Late Training CLI Output" width="700">
</div>

**Usage:**
```bash
# Run a single training job
late train config.yml

# Run with absolute path
late train /workspace/configs/my_training.yml

# The training will start immediately and show live output
```

**Note**: Memory is automatically cleared before and after the run, and all outputs are saved in `training_runs/` just like queue runs.

### Hyperparameter Sweeps (`late sweep`)

Run systematic hyperparameter sweeps to find optimal training configurations. Supports sweeping ANY parameter with early stopping and automatic report generation.

**Key Features:**
- Sweep any parameter (learning rate, batch size, LoRA rank, etc.)
- Override parameters for faster sweeps (e.g., shorter context length)
- Early stopping by percentage of epoch or step count
- Automatic Excel reports with loss graphs
- W&B integration with custom sweep IDs
- Add sweeps to queues for batch processing

**Command Line Usage:**
```bash
# Basic learning rate sweep with 25% epoch early stopping
late sweep config.yml --params learning_rate=1e-4,2e-4,3e-4 --percent-epoch 25

# Multiple parameters with overrides for efficiency
late sweep config.yml \
  --params learning_rate=1e-4,2e-4 lora.r=64,128 \
  --override max_seq_length=2048 \
  --percent-epoch 25

# Use custom sweep ID for W&B grouping
late sweep config.yml \
  --params learning_rate=1e-4,2e-4,3e-4 \
  --sweep-id "lr_search_v1" \
  --max-steps 100

# Add sweep to queue instead of running immediately
late sweep config.yml \
  --sweep-file lr_sweep.sweep \
  --add-to-queue overnight.qml
```

**Sweep File Format (.sweep):**
```yaml
# lr_sweep.sweep
sweep_id: "lr_optimization_v1"

sweep_parameters:
  learning_rate: [1e-5, 2e-5, 5e-5, 1e-4]
  lora.r: [64, 128]  # Can sweep nested parameters
  
overrides:
  max_seq_length: 2048  # Use shorter context for faster sweeps
  batch_size: 1         # Fix batch size for this sweep
  
early_stop:
  percent_epoch: 25  # OR use max_steps: 100
```

**Sweep Reports:**
After completion, an Excel report is generated with:
- Summary table of all configurations and results
- Best configuration highlighting
- Combined loss curves graph
- Individual graphs for top 5 configurations
- Saved in `sweep_runs/{sweep_id}/sweep_report_{sweep_id}.xlsx`

### Managing Training Queues (`late queue ...`)

A suite of commands to manage `.qml` batch files. By default, these commands operate on a `./queues/` directory.

-   **`late queue ls`**: List all available queues.
-   **`late queue create <name>.qml`**: Create a new, empty queue.
-   **`late queue add <queue>.qml <config>.yml`**: Add a training config path to a queue.
-   **`late queue delete <name>.qml`**: Delete a queue.
-   **`late queue start <name>.qml`**: **Start executing a queue**. The runner will process each job sequentially.
    - Automatically saves progress and can resume if interrupted
    - Use `--restart` to start from the beginning
    - Memory is cleared between each job
    - Sweep queues automatically generate reports upon completion

### Setting API Tokens (`late set`)

Securely store your Weights & Biases or Hugging Face tokens for automatic use in training runs.

**Usage:**
```bash
late set wandb YOUR_WANDB_API_KEY
late set hf_token YOUR_HUGGINGFACE_TOKEN
```

### Clearing GPU Memory (`late clear`)

Clears GPU VRAM, Python garbage collection, and PyTorch caches across all platforms (AMD ROCm, NVIDIA CUDA, CPU).

**Usage:**
```bash
# Clear all training memory
late clear
```

This command:
- Clears GPU VRAM (CUDA/ROCm automatically detected)
- Runs Python garbage collection
- Clears PyTorch caches
- Displays memory stats before/after with freed amounts

**When to use:**
- Between training runs to free up VRAM
- When encountering out-of-memory errors
- Before starting a new experiment

### Launching the Web Dashboard (`late serve`)

Starts a local web server to manage queues through a graphical interface.

**Usage:**
```bash
late serve --port 8080
```
Now open `http://localhost:8080` in your browser. If you run this on a remote server, you can forward the port or use a tool like ngrok to access it publicly.

### Merging LoRA Adapters (`late merge`)

Merge trained LoRA adapters with their base models and optionally upload to HuggingFace Hub.

**Usage:**
```bash
# Merge and upload to Hub
late merge /path/to/checkpoint-100 \
  --base-model Qwen/Qwen3-32B \
  --output username/merged-model

# Merge without uploading
late merge /path/to/checkpoint-100 \
  --base-model Qwen/Qwen3-32B \
  --output username/merged-model \
  --no-upload

# Specify local save path
late merge /path/to/checkpoint-100 \
  --base-model Qwen/Qwen3-32B \
  --output username/merged-model \
  --local-path ./my_merged_model

# Create private repository
late merge /path/to/checkpoint-100 \
  --base-model Qwen/Qwen3-32B \
  --output username/merged-model \
  --private

# Use config file
late merge /path/to/checkpoint-100 --config merge_config.yml
```

**Merge Config File Format:**
```yaml
adapter_path: "/path/to/checkpoint-100"
base_model: "Qwen/Qwen3-32B"
output_repo_id: "username/merged-model"
local_save_path: "./merged_output"
upload_to_hub: true
private_repo: false
hf_token: ""  # Optional, uses HF_TOKEN env var if not set
```

## 🌐 Web Dashboard

The web dashboard provides a user-friendly way to:
-   View all your training queues and the jobs within them.
-   Create new queues.
-   Delete existing queues.
-   Add new training jobs to a queue by providing the path to the config file.

<div align="center">
  <img src="assets/training_dashboard.png" alt="Late Training Dashboard" width="800">
</div>

*Note: Starting a queue is a CLI-only feature to ensure process stability.*

## 🎯 Loss Masking Strategies

Late supports two loss masking strategies for training:

### Full Loss Masking (Default)

**What it does:** Computes loss on the entire conversation (both user and assistant messages)

**Benefits:**
- ✅ Simpler and faster preprocessing
- ✅ Learns from complete conversations
- ✅ Good for most use cases
- ✅ Recommended for beginners

**Configuration:**
```yaml
# Default - can be omitted
loss_masking_strategy: "full"
```

### Assistant-Only Loss Masking

**What it does:** Masks user prompts from loss computation (sets to -100), only learns from assistant responses

**Benefits:**
- ✅ More targeted training (only learn assistant behavior)
- ✅ Prevents model from learning user patterns
- ✅ Traditional fine-tuning approach
- ✅ May be better for instruction-following tasks

**Trade-offs:**
- ⚠️ Slower preprocessing than full masking
- ⚠️ More complex implementation

**Configuration:**
```yaml
# Must be explicitly set
loss_masking_strategy: "assistant_only"
```

### Example Comparison

**Full Masking (Default):**
```yaml
base_model: "meta-llama/Llama-3.2-3B-Instruct"
dataset_name: "yahma/alpaca-cleaned"
loss_masking_strategy: "full"  # Simple preprocessing
# ... rest of config
```

**Assistant-Only Masking:**
```yaml
base_model: "meta-llama/Llama-3.2-3B-Instruct"
dataset_name: "yahma/alpaca-cleaned"
loss_masking_strategy: "assistant_only"  # Masked user prompts
# ... rest of config
```

See `examples/loss_masking/` for complete examples.

## ⚡ Unsloth Integration (Performance Boost)

Late now supports [Unsloth](https://github.com/unslothai/unsloth), an open-source library that significantly speeds up LLM fine-tuning while reducing memory usage. This integration is **optional** and provides:

### Benefits

- **2-5x Faster Training**: Optimized kernels for both AMD ROCm and NVIDIA CUDA GPUs
- **Lower Memory Usage**: More efficient memory management allows larger batch sizes
- **Zero Configuration**: Works out of the box with your existing configs
- **AMD GPU Optimized**: Automatically handles bitsandbytes instability on AMD GPUs
- **Backward Compatible**: Doesn't affect existing workflows - just add one flag

### How to Enable

Simply add `use_unsloth: true` to your training configuration:

```yaml
base_model: "unsloth/Llama-3.2-3B-Instruct"  # Unsloth-optimized models recommended
dataset_name: "mlabonne/FineTome-100k"
output_model_name: "username/my-fast-model"
output_dir: "./outputs/"
training_type: "lora"

# Enable Unsloth for 2-5x speedup
use_unsloth: true

# Rest of your config...
max_seq_length: 2048
batch_size: 4
gradient_accumulation: 4
epochs: 3
learning_rate: 2e-4

lora:
  r: 64
  lora_alpha: 128
```

### Installation

Unsloth requires special installation for AMD GPUs to ensure ROCm compatibility:

**For AMD GPUs (ROCm):**
```bash
# First, install Late with training dependencies
pip install late-training[training]

# Then, install Unsloth's AMD branch
pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"
```

**For NVIDIA GPUs (CUDA):**
```bash
# Install Late with training dependencies
pip install late-training[training]

# Then, install standard Unsloth
pip install unsloth
```

**Why the AMD-specific installation?**

The AMD branch of Unsloth includes:
- ROCm-compatible kernels
- Automatic bitsandbytes workarounds for HSA_STATUS_ERROR
- Optimizations for AMD GPU architectures (gfx90a, gfx942, gfx950, etc.)

### Recommended Models

For best performance, use Unsloth-optimized model variants:

- `unsloth/Llama-3.2-3B-Instruct` (instead of `meta-llama/Llama-3.2-3B-Instruct`)
- `unsloth/Meta-Llama-3-8B-Instruct`
- `unsloth/Qwen2.5-7B-Instruct`
- See [Unsloth Models](https://huggingface.co/unsloth) for more

### AMD GPU Notes

Unsloth automatically handles AMD-specific optimizations:

- **Bitsandbytes Compatibility**: Automatically uses 16-bit LoRA when `load_in_4bit=True` to avoid HSA_STATUS_ERROR exceptions
- **ROCm Kernel Support**: Optimized kernels for AMD architectures (gfx90a, gfx942, gfx950, etc.)
- **No Configuration Needed**: Works seamlessly on MI300X, RX 7900 XTX, and other AMD GPUs

### Performance Comparison

Typical speedups observed:

| Model Size | Without Unsloth | With Unsloth | Speedup |
|------------|----------------|--------------|---------|
| Llama 3.2 3B | 1.0x | 2.5x | 2.5x faster |
| Llama 3 8B | 1.0x | 3.2x | 3.2x faster |
| Qwen 2.5 7B | 1.0x | 2.8x | 2.8x faster |

*Benchmarks on AMD MI300X (192GB VRAM)*

### Example Configurations

See `examples/llama3/llama3.2_3b_lora.yml` for a complete example with Unsloth enabled.

## 🔔 Push Notifications with ntfy.sh

Get real-time notifications about your training runs on your phone or desktop!

### Setup

1. **Install ntfy app** on your phone ([iOS](https://apps.apple.com/us/app/ntfy/id1625396347) / [Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy))
2. **Choose a unique topic** (e.g., "mytraining-20250118")
3. **Subscribe** to your topic in the app
4. **Add to config:**

```yaml
ntfy_topic: "mytraining-20250118"
```

### Notifications Sent

- 🚀 Training start
- 💾 Checkpoint saves (every `save_steps`)
- ✅ Training completion
- 📤 Model upload success/failure
- 📂 Checkpoint resume events
- ❌ Error notifications

### Example Configuration

```yaml
base_model: "meta-llama/Llama-3.2-3B-Instruct"
dataset_name: "yahma/alpaca-cleaned"
output_model_name: "username/my-model"
output_dir: "./outputs/"
training_type: "lora"

# Enable notifications
ntfy_topic: "my-training-notifications"

# Checkpoints trigger notifications
save_steps: 50  # Notification every 50 steps

# ... rest of config
```

### Testing Notifications

Send a test notification:
```bash
curl -d "Test from Late!" ntfy.sh/your-topic-name
```

See `examples/with_notifications.yml` for a complete example.

## 📂 Example Training Configurations

<details>
<summary><b>Example 1: Full Supervised Fine-Tuning (SFT)</b></summary>

This configuration fine-tunes all the parameters of the base model.

**`sft_config.yml`**
```yaml
base_model: "Qwen/Qwen2-1.5B-Instruct"
dataset_name: "Tesslate/UIGENT"
output_model_name: "your-hf-username/UIGENT-1.5B-SFT"
output_dir: "/scratch/outputs/sft/"
training_type: "sft"
max_seq_length: 4096
batch_size: 2
gradient_accumulation: 8  # Effective batch size: 16 (2 * 8)
gradient_checkpointing: true  # Reduce VRAM usage (default: true)
epochs: 1
learning_rate: 2.0e-5
report_to_wandb: true
cache_dir: "~/.cache/late/models"  # Model cache directory
```

</details>

<details>
<summary><b>Example 2: LoRA Fine-Tuning</b></summary>

This configuration uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to train only a small number of adapter weights, which is much faster and more memory-efficient.

**`lora_config.yml`**
```yaml
base_model: "Qwen/Qwen2-7B-Instruct"
dataset_name: "smirki/UIGENT-9-6-25"
output_model_name: "your-hf-username/UIGENT-7B-Lora"
output_dir: "/scratch/outputs/lora/"
training_type: "lora" # Set to 'lora' to enable PEFT

# Training Hyperparameters
max_seq_length: 8192
batch_size: 1
gradient_accumulation: 16
epochs: 2
learning_rate: 2.0e-4 # Higher learning rate is common for LoRA

# Memory Optimization
gradient_checkpointing: true  # Highly recommended for large models

# LoRA Specific Config
lora:
 r: 128
 lora_alpha: 256
 target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"

# Control Flags
report_to_wandb: true
upload_to_hub: true
```

</details>

### 🦙 Pre-configured Examples

Check out the [`examples/`](examples/) directory for ready-to-use configurations:

- **Llama 3 Examples** - Full configurations for Llama 3 8B and 3.2B models
- **Sweep Templates** - Pre-built sweep configurations for common use cases
- **Production Queues** - Example queue setups for real workflows

## 📋 Complete Configuration Reference

### Required Parameters
- `base_model`: HuggingFace model ID or path
- `dataset_name`: HuggingFace dataset ID (must have "messages" field)
- `output_model_name`: Name for uploaded model (format: "username/model-name")
- `output_dir`: Local directory to save model
- `training_type`: Either "sft" (full fine-tuning) or "lora" (parameter-efficient)
- `max_seq_length`: Maximum sequence length for training

### Training Parameters
- `batch_size`: Per-device batch size (default: 1)
- `gradient_accumulation`: Gradient accumulation steps (default: 16)
- `epochs`: Number of training epochs (default: 1)
- `learning_rate`: Learning rate (default: 2e-5)
- `lr_scheduler_type`: LR scheduler type (default: "linear")
- `optim`: Optimizer (default: "adamw_torch_fused" - optimized for ROCm)
- `save_steps`: Save checkpoint every N steps (default: 50)

### Performance Optimization (NEW)
- `use_unsloth`: Enable Unsloth for 2-5x faster training (default: false)
  - Recommended for both AMD and NVIDIA GPUs
  - Works best with Unsloth-optimized models (e.g., `unsloth/Llama-3.2-3B-Instruct`)
  - Automatically handles AMD GPU optimizations

### Memory & Performance
- `gradient_checkpointing`: Enable gradient checkpointing to reduce VRAM (default: true)
- `torch_compile`: Enable torch.compile for faster training (default: true)
- `tf32`: Enable TF32 mode for matrix operations (default: true)
- `cache_dir`: Model cache directory (default: "~/.cache/late/models")

### LoRA Configuration (when training_type: "lora")
```yaml
lora:
  r: 128                    # LoRA rank
  lora_alpha: 256          # LoRA alpha scaling parameter
  target_modules:          # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

### Loss Masking Strategy (NEW)
- `loss_masking_strategy`: Choose loss computation strategy (default: "full")
  - `"full"`: Compute loss on entire conversation (default, simpler, faster)
  - `"assistant_only"`: Mask user prompts from loss (-100 tokens)

### Notifications & Tokens (NEW)
- `ntfy_topic`: ntfy.sh topic for push notifications (optional, default: none)
- `hf_token`: HuggingFace API token (optional, uses HF_TOKEN env var if not set)

### Logging & Upload
- `report_to_wandb`: Enable Weights & Biases logging (default: false)
- `upload_to_hub`: Upload to HuggingFace Hub after training (default: false)

### Complete Example Configuration
```yaml
# Model & Dataset
base_model: "meta-llama/Llama-3.2-3B-Instruct"
dataset_name: "your-dataset/chat-format"
output_model_name: "your-username/model-finetuned"
output_dir: "/workspace/outputs/my-model/"

# Training Setup
training_type: "lora"  # or "sft" for full fine-tuning
max_seq_length: 8192

# Loss Masking Strategy (NEW - optional, defaults to "full")
loss_masking_strategy: "full"  # or "assistant_only"

# Hyperparameters
batch_size: 1
gradient_accumulation: 16  # Effective batch size: 16
epochs: 3
learning_rate: 2e-4
lr_scheduler_type: "cosine"
optim: "adamw_torch_fused"
save_steps: 100

# Memory & Performance
gradient_checkpointing: true
torch_compile: true
tf32: true
cache_dir: "~/.cache/late/models"

# LoRA Config (only if training_type: "lora")
lora:
  r: 128
  lora_alpha: 256
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Notifications & Tokens (NEW - optional)
ntfy_topic: "my-training-runs"  # Push notifications
hf_token: ""  # Or uses HF_TOKEN env var

# Logging
report_to_wandb: true
upload_to_hub: true
```

## 🔧 Training Run Management

Late automatically manages your training runs:

- **Run Directory**: Each training run creates a timestamped directory in `training_runs/`
- **Config Preservation**: The YAML config is saved as JSON for reproducibility
- **Training Script**: The generated Python script is saved for debugging
- **Logs**: Full training output is saved to `training.log`
- **Memory Management**: VRAM is automatically cleared between queue runs

Example run directory structure:
```
training_runs/
└── UIGENT-7B-Lora_lora_20240315_143022/
    ├── config.json          # Training configuration
    ├── training_script.py   # Generated training script
    └── training.log         # Full output log
```

## 💡 Example Workflow (End-to-End)

Here’s how to go from a fresh server to a running batch of experiments.

```bash
# 1. Create a dedicated Python virtual environment
python3 -m venv rocm_trainer_env
source rocm_trainer_env/bin/activate

# 2. Install the 'late' library inside the venv
# (Assuming you've cloned the repo)
pip install -e .

# 3. Patch this new environment. This will take a while!
# No need for --venv flag since we are already inside it.
late patch

# 4. Set your API keys
late set wandb <your_key>
late set hf_token <your_key>

# 5. Create YAML config files for your experiments
# (e.g., `configs/sft_qwen.yml`, `configs/lora_llama.yml`)

# 6. Create a new training queue
late queue create nightly_runs.qml

# 7. Add your experiments to the queue
late queue add nightly_runs.qml configs/sft_qwen.yml
late queue add nightly_runs.qml configs/lora_llama.yml

# 8. Start the queue and let it run!
late queue start nightly_runs.qml
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/TesslateAI/Late.git
cd Late

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for the ROCm community
- Powered by [HuggingFace Transformers](https://huggingface.co/transformers/)
- Flash Attention implementation from [ROCm/flash-attention](https://github.com/ROCm/flash-attention)

## 📞 Support

- 📧 Email: support@tesslate.ai
- 🐛 Issues: [GitHub Issues](https://github.com/TesslateAI/Late/issues)
- 💬 Discord: [Join our community](https://discord.gg/tesslate)

---

<div align="center">
  <p>
    <a href="https://tesslate.ai">
      <img src="assets/tesslate-logo.png" alt="TesslateAI" width="150">
    </a>
  </p>
  <p>
    Made with 🔥 by <a href="https://tesslate.ai">TesslateAI</a>
  </p>
</div>
