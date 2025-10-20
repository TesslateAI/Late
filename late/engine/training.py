import subprocess
import sys
import yaml
import json
import os
import gc
import torch
from datetime import datetime
from pathlib import Path
from .config import load_tokens

def clear_memory():
    """Clear CPU and GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_training_job(config_path: str):
    """
    Runs a training job by generating a Python script from a YAML config
    and executing it.
    """
    print(f"\n[START] Launching training job for: {config_path}\n{'='*50}")

    # Clear memory before starting
    clear_memory()

    # Load tokens into environment for the subprocess
    load_tokens()

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('output_model_name', 'unknown').split('/')[-1]

    # Check if this is a sweep run
    if '_sweep_metadata' in config:
        sweep_meta = config['_sweep_metadata']
        run_name = f"{model_name}_{sweep_meta['sweep_id']}_run{sweep_meta['sweep_index']}"
    else:
        run_name = f"{model_name}_{config.get('training_type', 'sft')}_{timestamp}"

    run_dir = Path("training_runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_save_path = run_dir / "config.json"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    script_content = generate_training_script(config)

    # Save script with unique name
    script_path = run_dir / "training_script.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Config saved to: {config_save_path}")

    # Execute the generated script
    process = subprocess.Popen([sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Also save output to log file
    log_path = run_dir / "training.log"
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)

    process.wait()

    # Clear memory after completion
    clear_memory()

    if process.returncode == 0:
        print(f"\n{'='*50}\n[SUCCESS] Training job for '{config_path}' completed successfully.")
        print(f"[INFO] Results saved in: {run_dir}")
    else:
        print(f"\n{'='*50}\n[FAILED] Training job for '{config_path}' failed with exit code {process.returncode}.")
        print(f"[INFO] Logs saved in: {run_dir}")

def generate_training_script(config: dict) -> str:
    """Generates the full Python training script based on the config."""

    # This function is large because it contains the full, dynamic script.
    # It directly translates the user-provided notebook cells into a runnable script.

    # Determine training type (SFT or LoRA)
    is_lora = config.get('training_type', '').lower() == 'lora'

    # Determine loss masking strategy (default: "full")
    loss_strategy = config.get('loss_masking_strategy', 'full')

    # Check if Unsloth should be used
    use_unsloth = config.get('use_unsloth', False)

    # Build the script string
    script = f"""
import os
import torch
import logging
import requests
{'from unsloth import FastLanguageModel' if use_unsloth else '# Unsloth not enabled'}
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set HF Token from config or environment
hf_token = "{config.get('hf_token', '')}" or os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# Detect available hardware
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    if hasattr(torch.version, 'hip') and 'rocm' in torch.version.hip:
        logger.info(f"[OK] Detected AMD ROCm GPU: {{device_name}}")
        device_type = "rocm"
    else:
        logger.info(f"[OK] Detected NVIDIA CUDA GPU: {{device_name}}")
        device_type = "cuda"
else:
    logger.info("[INFO] No GPU detected. Running on CPU. Training will be slower.")
    device_type = "cpu"

config = {config}

# Extract sweep metadata if present
sweep_metadata = config.pop('_sweep_metadata', None)
wandb_run_name_override = config.pop('_wandb_run_name', None)

# --- ntfy Notification Utilities ---
def send_ntfy(message, title="Late Trainer"):
    \"\"\"Sends a notification to the configured ntfy.sh topic.\"\"\"
    ntfy_topic = config.get('ntfy_topic', '')
    if not ntfy_topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{{ntfy_topic}}",
            data=message.encode('utf-8'),
            headers={{"Title": title, "Priority": "default", "Tags": "rocket"}}
        )
        logger.info(f"[INFO] Notification sent to ntfy topic: {{ntfy_topic}}")
    except Exception as e:
        logger.error(f"[WARN] Failed to send ntfy notification: {{e}}")

class NtfyCheckpointCallback(TrainerCallback):
    \"\"\"A custom TrainerCallback to send ntfy notifications on checkpoint saves.\"\"\"
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{{state.global_step}}")
        logger.info(f"[SAVE] Checkpoint saved to {{checkpoint_path}}. Sending notification...")
        send_ntfy(
            f"[OK] Checkpoint saved at step {{state.global_step}}.",
            title="Checkpoint Saved"
        )
        return control

# --- 2. W&B Setup ---
if config.get('report_to_wandb', False):
    os.environ.setdefault("WANDB_PROJECT", "Late-Training-Runs")
    if sweep_metadata:
        # Add sweep tags to W&B
        os.environ["WANDB_TAGS"] = f"{{sweep_metadata['sweep_id']}}"
        logger.info(f"W&B reporting enabled for sweep: {{sweep_metadata['sweep_id']}}")
    else:
        logger.info("W&B reporting is enabled.")

# --- 3. Model and Tokenizer Loading ---
logger.info(f"Loading base model: {{config['base_model']}}")

# Configure model loading based on available hardware
if device_type == "cpu":
    dtype = torch.float32  # CPU doesn't support bfloat16 as efficiently
    device_map = "cpu"
    attn_impl = "eager"  # Flash attention not available on CPU
    logger.info("[INFO] Using CPU with float32 precision")
else:
    dtype = torch.bfloat16
    device_map = {{'': torch.cuda.current_device()}}
    attn_impl = "flash_attention_2"
    logger.info(f"[INFO] Using GPU with bfloat16 precision and Flash Attention 2")

"""

    # Add model loading based on whether Unsloth is used
    if use_unsloth:
        script += """
# Using Unsloth for optimized training
# Note: Unsloth handles bitsandbytes instability on AMD GPUs by automatically
# using 16-bit LoRA when load_in_4bit=True to avoid HSA_STATUS_ERROR exceptions
logger.info("[INFO] Using Unsloth for accelerated training")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['base_model'],
    max_seq_length = config['max_seq_length'],
    dtype = dtype,
    load_in_4bit = True,
)

# Unsloth handles tokenizer setup - using as-is
"""
    else:
        script += """
model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    torch_dtype=dtype,
    attn_implementation=attn_impl,
    use_cache=False if config.get('gradient_checkpointing', True) else True,
    device_map=device_map,
    cache_dir=os.path.expanduser(config.get('cache_dir', '~/.cache/late/models')),
)

# Enable gradient checkpointing if specified
if config.get('gradient_checkpointing', True):
    model.gradient_checkpointing_enable()
    logger.info("✓ Gradient checkpointing enabled")

tokenizer = AutoTokenizer.from_pretrained(config['base_model'], cache_dir=os.path.expanduser(config.get('cache_dir', '~/.cache/late/models')))

# Set up tokenizer padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
"""

    script += """
# --- 4. LoRA Configuration (if applicable) ---
"""
    if is_lora:
        if use_unsloth:
            script += """
logger.info("Applying LoRA configuration with Unsloth...")
lora_config_data = config.get('lora', {})
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_config_data.get('r', 128),
    target_modules = lora_config_data.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    lora_alpha = lora_config_data.get('lora_alpha', 256),
    lora_dropout = 0,  # Unsloth optimizes with 0 dropout
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth's optimized gradient checkpointing
    random_state = 3407,
    max_seq_length = config['max_seq_length'],
)
model.print_trainable_parameters()
"""
        else:
            script += """
logger.info("Applying LoRA configuration...")
lora_config_data = config.get('lora', {})
peft_config = LoraConfig(
    r=lora_config_data.get('r', 128),
    lora_alpha=lora_config_data.get('lora_alpha', 256),
    target_modules=lora_config_data.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""

    # Add dataset preprocessing based on loss masking strategy
    if loss_strategy == 'full':
        if use_unsloth:
            # Use Unsloth's dataset formatting methods
            script += f"""
# --- 5. Dataset Loading and Preprocessing (Simple Unsloth approach) ---
logger.info(f"Loading dataset: {{config['dataset_name']}}")
from datasets import load_dataset
dataset = load_dataset(config['dataset_name'], split="train")
tokenizer.model_max_length = config['max_seq_length']

def format_for_training(example):
    \"\"\"Simple formatting using tokenizer's built-in chat template\"\"\"
    # Support both 'messages' and 'conversations' field names
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        # Convert from 'from'/'value' format to 'role'/'content' format
        conversations = example["conversations"]
        messages = []
        for msg in conversations:
            role_map = {{"human": "user", "gpt": "assistant", "system": "system"}}
            role = role_map.get(msg.get("from", ""), msg.get("from", "user"))
            content = msg.get("value", "")
            messages.append({{"role": role, "content": content}})
    else:
        raise ValueError(f"No messages or conversations field found. Available keys: {{list(example.keys())}}")
    return {{"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}}

# Filter out examples with None or empty messages
def has_valid_messages(example):
    msgs = None
    if "messages" in example:
        msgs = example["messages"]
    elif "conversations" in example:
        msgs = example["conversations"]
    return msgs is not None and isinstance(msgs, list) and len(msgs) > 0

dataset = dataset.filter(has_valid_messages)
processed_dataset = dataset.map(format_for_training, remove_columns=list(dataset.features))
logger.info(f"✓ Dataset formatted for Unsloth. Total examples: {{len(processed_dataset)}}")
"""
        else:
            # Original formatting for standard training
            script += f"""
# --- 5. Dataset Loading and Preprocessing (FULL Loss Masking - DEFAULT) ---
logger.info(f"Loading dataset: {{config['dataset_name']}}")
dataset = load_dataset(config['dataset_name'], split="train")
tokenizer.model_max_length = config['max_seq_length']

def format_for_training(example):
    \"\"\"Simple formatting using chat template - computes loss on full conversation\"\"\"
    # Support both 'messages' and 'conversations' field names
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        # Convert from 'from'/'value' format to 'role'/'content' format
        conversations = example["conversations"]
        messages = []
        for msg in conversations:
            role_map = {{"human": "user", "gpt": "assistant", "system": "system"}}
            role = role_map.get(msg.get("from", ""), msg.get("from", "user"))
            content = msg.get("value", "")
            messages.append({{"role": role, "content": content}})
    else:
        raise ValueError(f"No messages or conversations field found. Available keys: {{list(example.keys())}}")
    return {{"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}}

logger.info("Applying 'full' loss masking strategy (compute loss on entire conversation)...")

# Filter out examples with None or empty messages
def has_valid_messages(example):
    msgs = None
    if "messages" in example:
        msgs = example["messages"]
    elif "conversations" in example:
        msgs = example["conversations"]
    return msgs is not None and isinstance(msgs, list) and len(msgs) > 0

dataset = dataset.filter(has_valid_messages)
logger.info(f"Filtered dataset to {{len(dataset)}} examples with valid messages")

processed_dataset = dataset.map(format_for_training, remove_columns=list(dataset.features))
logger.info(f"✓ Dataset formatted. Total examples: {{len(processed_dataset)}}")
"""
    else:
        # Assistant-only - mask user prompts from loss
        script += f"""
# --- 5. Dataset Loading and Preprocessing (ASSISTANT-ONLY Loss Masking) ---
logger.info(f"Loading dataset: {{config['dataset_name']}}")
dataset = load_dataset(config['dataset_name'], split="train")
tokenizer.model_max_length = config['max_seq_length']

def preprocess_for_assistant_loss(examples, tokenizer):
    \"\"\"Masks user prompts from loss calculation - only compute loss on assistant responses\"\"\"
    all_input_ids, all_labels = [], []
    # Support both 'messages' and 'conversations' field names
    if "messages" in examples:
        conversation_field = "messages"
    elif "conversations" in examples:
        conversation_field = "conversations"
    else:
        raise ValueError(f"No messages or conversations field found")
    for messages in examples[conversation_field]:
        current_input_ids, current_labels = [], []
        if tokenizer.bos_token:
            bos_tokens = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
            current_input_ids.extend(bos_tokens)
            current_labels.extend([-100] * len(bos_tokens))

        for message in messages:
            formatted_turn = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            tokenized_turn = tokenizer(formatted_turn, add_special_tokens=False)
            turn_input_ids = tokenized_turn["input_ids"]

            turn_labels = list(turn_input_ids) if message['role'] == 'assistant' else [-100] * len(turn_input_ids)

            current_input_ids.extend(turn_input_ids)
            current_labels.extend(turn_labels)

        if tokenizer.eos_token:
            eos_tokens = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
            current_input_ids.extend(eos_tokens)
            current_labels.extend(eos_tokens) # Learn to predict EOS

        max_len = tokenizer.model_max_length
        all_input_ids.append(current_input_ids[:max_len])
        all_labels.append(current_labels[:max_len])

    return {{"input_ids": all_input_ids, "labels": all_labels}}

logger.info("Applying 'assistant_only' loss masking strategy (mask user prompts from loss)...")
processed_dataset = dataset.map(
    preprocess_for_assistant_loss,
    fn_kwargs={{"tokenizer": tokenizer}},
    batched=True,
    batch_size=100,
    remove_columns=dataset.column_names,
)
logger.info(f"✓ Dataset preprocessed. Total examples: {{len(processed_dataset)}}")
"""

    script += f"""
# --- 6. Trainer Configuration ---
logger.info("Configuring SFTTrainer...")

# Calculate effective batch size
per_device_batch = config.get('batch_size', 1)
gradient_accumulation = config.get('gradient_accumulation', 16)
effective_batch_size = per_device_batch * gradient_accumulation
logger.info(f"Effective batch size: {{effective_batch_size}} ({{per_device_batch}} * {{gradient_accumulation}} accumulation steps)")

# Configure precision and optimization based on hardware
training_kwargs = {{
    'output_dir': config['output_dir'],
    'num_train_epochs': config.get('epochs', 1),
    'per_device_train_batch_size': per_device_batch,
    'gradient_accumulation_steps': gradient_accumulation,
    'learning_rate': float(config.get('learning_rate', 2e-5)),
    'lr_scheduler_type': config.get('lr_scheduler_type', 'linear'),
    'max_length': config['max_seq_length'],
    'gradient_checkpointing': config.get('gradient_checkpointing', True),
    'logging_steps': 10,
    'save_steps': config.get('save_steps', 50),
    'save_strategy': "steps",
    'save_total_limit': 5,
    'report_to': ["wandb"] if config.get('report_to_wandb') else "none",
    'run_name': wandb_run_name_override or f"{{config['output_model_name'].split('/')[-1]}}-{{config.get('training_type', 'sft')}}",
}}

# Add hardware-specific optimizations
if device_type != "cpu":
    # GPU-specific optimizations
    training_kwargs['bf16'] = True
    # Only set tf32 if explicitly enabled (not available on AMD GPUs)
    if config.get('tf32', False):
        training_kwargs['tf32'] = True
    training_kwargs['torch_compile'] = config.get('torch_compile', True)
    training_kwargs['optim'] = config.get('optim', 'adamw_torch_fused')
else:
    # CPU fallback - no bf16, tf32, or torch_compile
    training_kwargs['fp16'] = False
    training_kwargs['optim'] = 'adamw_torch'
    logger.info("[INFO] Using CPU-compatible optimizer (adamw_torch)")

training_args = SFTConfig(**training_kwargs)
"""

    # Add dataset_text_field for full loss masking
    if loss_strategy == 'full':
        script = script.replace(
            "training_args = SFTConfig(**training_kwargs)",
            "training_kwargs['dataset_text_field'] = 'text'\ntraining_args = SFTConfig(**training_kwargs)"
        )

    script += f"""
# Data collator (only needed for assistant_only strategy)
"""
    if loss_strategy != 'full':
        script += f"""data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
"""

    script += f"""
# Prepare callbacks
callbacks = []
if config.get('ntfy_topic'):
    callbacks.append(NtfyCheckpointCallback())

# Create trainer
trainer_kwargs = {{
    'model': model,
    'processing_class': tokenizer,
    'args': training_args,
    'train_dataset': processed_dataset,
    'callbacks': callbacks,
}}

"""


    script += f"""# Add data_collator only for assistant_only strategy
"""
    if loss_strategy != 'full':
        script += f"""trainer_kwargs['data_collator'] = data_collator
"""

    script += f"""
trainer = SFTTrainer(**trainer_kwargs)

# --- 7. Checkpoint Resume Logic ---
last_checkpoint = None
if os.path.isdir(config['output_dir']) and not config.get('force_restart', False):
    checkpoints = sorted(
        [d for d in os.listdir(config['output_dir']) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0
    )
    if checkpoints:
        last_checkpoint = os.path.join(config['output_dir'], checkpoints[-1])
        logger.info(f"[RESUME] Resuming training from checkpoint: {{last_checkpoint}}")
        send_ntfy(f"Resuming training from checkpoint: {{checkpoints[-1]}}")

# --- 8. Start Training ---
logger.info("[START] Starting training...")
send_ntfy(f"[START] Starting training job for {{config['base_model']}}")

# Handle early stopping for sweeps
if sweep_metadata and 'early_stop' in sweep_metadata:
    early_stop = sweep_metadata['early_stop']

    if 'percent_epoch' in early_stop:
        # Calculate number of steps for percentage of epoch
        percent = early_stop['percent_epoch']
        total_steps = len(processed_dataset) // (config.get('batch_size', 1) * config.get('gradient_accumulation', 16))
        max_steps = int(total_steps * percent / 100)
        training_args.max_steps = max_steps
        logger.info(f"Early stopping at {{percent}}% of epoch ({{max_steps}} steps)")

    elif 'max_steps' in early_stop:
        # Use explicit step count
        training_args.max_steps = early_stop['max_steps']
        logger.info(f"Early stopping at {{early_stop['max_steps']}} steps")

    # Update trainer with new args
    trainer.args = training_args

# Train with resume capability
if last_checkpoint and not config.get('force_restart', False):
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

logger.info("[SUCCESS] Training complete.")
send_ntfy("[SUCCESS] Training complete! Starting final save and upload.", title="Training Finished")

# Save training history for sweep analysis
if sweep_metadata:
    import json
    history = {{
        'loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log],
        'steps': [log['step'] for log in trainer.state.log_history if 'loss' in log],
        'sweep_params': sweep_metadata['sweep_params']
    }}
    history_path = config['output_dir'] + '/training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f)

# --- 9. Save and Upload ---
logger.info(f"[SAVE] Saving final model to {{config['output_dir']}}...")
trainer.save_model(config['output_dir'])
tokenizer.save_pretrained(config['output_dir'])

if config.get('upload_to_hub', False):
    try:
        logger.info(f"📤 Uploading model to Hugging Face Hub: {{config['output_model_name']}}")
        api = HfApi(token=hf_token if hf_token else None)
        api.create_repo(repo_id=config['output_model_name'], exist_ok=True, private=False)
        api.upload_folder(
            folder_path=config['output_dir'],
            repo_id=config['output_model_name'],
            commit_message="Training run with 'Late' library",
        )
        logger.info(f"[OK] Model successfully uploaded to https://huggingface.co/{{config['output_model_name']}}")
        send_ntfy(
            f"[SUCCESS] Model uploaded to HF Hub: {{config['output_model_name']}}",
            title="Upload Complete"
        )
    except Exception as e:
        logger.error(f"[WARN] Error uploading to Hugging Face Hub: {{e}}")
        send_ntfy(f"[ERROR] Model upload failed: {{e}}", title="Error")

"""
    return script
