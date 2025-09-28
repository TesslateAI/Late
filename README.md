# Late
> A powerful toolkit for streamlining and scheduling machine learning training workflows on ROCm servers.

`Late` is a Python library that provides a unified interface for managing the entire lifecycle of training large models on AMD GPUs. It combines a powerful **Command-Line Interface (CLI)**, a **batch job runner**, and a user-friendly **web dashboard** to simplify environment setup, job scheduling, and execution.

---

## ✨ Key Features

-   **Automated Environment Patching**: A single command (`late patch`) installs and configures Flash Attention for ROCm, targeting either your global environment or a specific Python virtual environment. By default uses pre-built wheels, with optional source building.
-   **Declarative Training Jobs**: Define all aspects of your training runs—from model choice to hyperparameters—in simple, readable YAML configuration files.
-   **Batch Queue Management**: Group multiple training configs into `.qml` queue files to run them sequentially. Perfect for running a series of experiments overnight.
-   **Versatile CLI**: Manage every aspect of your workflow from the terminal, including patching, training, and queue management.
-   **Web Dashboard**: Launch a web server (`late serve`) to visually create, manage, and monitor your training queues from any browser.
-   **Built for ROCm**: Optimized with defaults like `adamw_torch_fused` and `bfloat16` to get the best performance out of AMD hardware.

## 📦 Installation

Clone the repository and install the package in editable mode using pip. This will make the `late` command available in your shell.

```bash
git clone https://github.com/your-username/late.git
cd late
pip install -e .
```

## 🚀 Core Concepts

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

### Environment Patching (`late patch`)

Prepares a Python environment for ROCm training. It automatically installs Flash Attention and other core ML libraries. By default, Flash Attention is installed from a pre-built wheel, but you can build from source using the `--source` flag.

**Usage:**
```bash
# Patch the current environment for a gfx942 GPU (uses pre-built wheel)
late patch --arch gfx942

# Build Flash Attention from source
late patch --arch gfx942 --source

# Create a venv and patch it specifically (recommended)
python3 -m venv my_env
late patch --venv ./my_env

# Patch venv and build from source
late patch --venv ./my_env --source
```

### Running a Single Training Job (`late train`)

Immediately starts a training job from a single configuration file.

**Usage:**
```bash
late train /path/to/your/sft_config.yml
```

### Managing Training Queues (`late queue ...`)

A suite of commands to manage `.qml` batch files. By default, these commands operate on a `./queues/` directory.

-   **`late queue ls`**: List all available queues.
-   **`late queue create <name>.qml`**: Create a new, empty queue.
-   **`late queue add <queue>.qml <config>.yml`**: Add a training config path to a queue.
-   **`late queue delete <name>.qml`**: Delete a queue.
-   **`late queue start <name>.qml`**: **Start executing a queue**. The runner will process each job sequentially.

### Setting API Tokens (`late set`)

Securely store your Weights & Biases or Hugging Face tokens for automatic use in training runs.

**Usage:**
```bash
late set wandb YOUR_WANDB_API_KEY
late set hf_token YOUR_HUGGINGFACE_TOKEN
```

### Launching the Web Dashboard (`late serve`)

Starts a local web server to manage queues through a graphical interface.

**Usage:**
```bash
late serve --port 8080
```
Now open `http://localhost:8080` in your browser. If you run this on a remote server, you can forward the port or use a tool like ngrok to access it publicly.

## 🌐 Web Dashboard

The web dashboard provides a user-friendly way to:
-   View all your training queues and the jobs within them.
-   Create new queues.
-   Delete existing queues.
-   Add new training jobs to a queue by providing the path to the config file.

![Web Dashboard Screenshot](https://i.imgur.com/placeholder.png)  <!-- Placeholder for a future screenshot -->
*Note: Starting a queue is a CLI-only feature to ensure process stability.*

## 📝 Example Training Configurations

### Example 1: Full Supervised Fine-Tuning (SFT)

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
gradient_accumulation: 8
epochs: 1
learning_rate: 2.0e-5
report_to_wandb: true
```

### Example 2: LoRA Fine-Tuning

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