
import os
import torch
import logging
import requests
import sys
# Unsloth not enabled
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set HF Token from config or environment
hf_token = "" or os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
else:
    logger.error("Huggingface token invalid or not found.")
    sys.exit(1)

# Detect available hardware
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name('cuda')
    if hasattr(torch.version, 'hip') and torch.version.hip and 'rocm' in torch.version.hip:
        logger.info(f"[OK] Detected AMD ROCm GPU: {device_name}")
        device_type = "rocm"
    else:
        logger.info(f"[OK] Detected NVIDIA CUDA GPU: {device_name}")
        device_type = "cuda"
else:
    logger.info("[INFO] No GPU detected. Running on CPU. Training will be slower.")
    device_type = "cpu"

config = {'base_model': 'mistralai/Mistral-7B-Instruct-v0.3', 'dataset_name': 'alpaca-cleaned', 'output_model_name': 'my-first-lora', 'output_dir': './outputs/quick-test/', 'training_type': 'lora', 'max_seq_length': 2048, 'batch_size': 4, 'gradient_accumulation': 4, 'epochs': 1, 'learning_rate': '2e-4', 'lora': {'r': 32, 'lora_alpha': 64}}

# Extract sweep metadata if present
sweep_metadata = config.pop('_sweep_metadata', None)
wandb_run_name_override = config.pop('_wandb_run_name', None)

# --- ntfy Notification Utilities ---
def send_ntfy(message, title="Late Trainer"):
    """Sends a notification to the configured ntfy.sh topic."""
    ntfy_topic = config.get('ntfy_topic', '')
    if not ntfy_topic:
        return
    try:
        requests.post(
            f"https://ntfy.sh/{ntfy_topic}",
            data=message.encode('utf-8'),
            headers={"Title": title, "Priority": "default", "Tags": "rocket"}
        )
        logger.info(f"[INFO] Notification sent to ntfy topic: {ntfy_topic}")
    except Exception as e:
        logger.error(f"[WARN] Failed to send ntfy notification: {e}")

class NtfyCheckpointCallback(TrainerCallback):
    """A custom TrainerCallback to send ntfy notifications on checkpoint saves."""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        logger.info(f"[SAVE] Checkpoint saved to {checkpoint_path}. Sending notification...")
        send_ntfy(
            f"[OK] Checkpoint saved at step {state.global_step}.",
            title="Checkpoint Saved"
        )
        return control

# --- 2. W&B Setup ---
if config.get('report_to_wandb', False):
    os.environ.setdefault("WANDB_PROJECT", "Late-Training-Runs")
    if sweep_metadata:
        # Add sweep tags to W&B
        os.environ["WANDB_TAGS"] = f"{sweep_metadata['sweep_id']}"
        logger.info(f"W&B reporting enabled for sweep: {sweep_metadata['sweep_id']}")
    else:
        logger.info("W&B reporting is enabled.")

# --- 3. Model and Tokenizer Loading ---
logger.info(f"Loading base model: {config['base_model']}")

# Configure model loading based on available hardware
if device_type == "cpu":
    dtype = torch.float32  # CPU doesn't support bfloat16 as efficiently
    device_map = "cpu"
    attn_impl = "eager"  # Flash attention not available on CPU
    logger.info("[INFO] Using CPU with float32 precision")
else:
    dtype = torch.bfloat16
    device_map = {'': torch.cuda.current_device()}
    attn_impl = "flash_attention_2"
    logger.info(f"[INFO] Using GPU with bfloat16 precision and Flash Attention 2")


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

# --- 4. LoRA Configuration (if applicable) ---

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

# --- 5. Dataset Loading and Preprocessing (FULL Loss Masking - DEFAULT) ---
logger.info(f"Loading dataset: {config['dataset_name']}")
dataset = load_dataset(config['dataset_name'], split="train")
tokenizer.model_max_length = config['max_seq_length']

def format_for_training(example):
    """Simple formatting using chat template - computes loss on full conversation"""
    # Support both 'messages' and 'conversations' field names
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        # Convert from 'from'/'value' format to 'role'/'content' format
        conversations = example["conversations"]
        messages = []
        for msg in conversations:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(msg.get("from", ""), msg.get("from", "user"))
            content = msg.get("value", "")
            messages.append({"role": role, "content": content})
    else:
        raise ValueError(f"No messages or conversations field found. Available keys: {list(example.keys())}")
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

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
logger.info(f"Filtered dataset to {len(dataset)} examples with valid messages")

processed_dataset = dataset.map(format_for_training, remove_columns=list(dataset.features))
logger.info(f"✓ Dataset formatted. Total examples: {len(processed_dataset)}")

# --- 6. Trainer Configuration ---
logger.info("Configuring SFTTrainer...")

# Calculate effective batch size
per_device_batch = config.get('batch_size', 1)
gradient_accumulation = config.get('gradient_accumulation', 16)
effective_batch_size = per_device_batch * gradient_accumulation
logger.info(f"Effective batch size: {effective_batch_size} ({per_device_batch} * {gradient_accumulation} accumulation steps)")

# Configure precision and optimization based on hardware
training_kwargs = {
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
    'run_name': wandb_run_name_override or f"{config['output_model_name'].split('/')[-1]}-{config.get('training_type', 'sft')}",
}

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

training_kwargs['dataset_text_field'] = 'text'
training_args = SFTConfig(**training_kwargs)

# Data collator (only needed for assistant_only strategy)

# Prepare callbacks
callbacks = []
if config.get('ntfy_topic'):
    callbacks.append(NtfyCheckpointCallback())

# Create trainer
trainer_kwargs = {
    'model': model,
    'processing_class': tokenizer,
    'args': training_args,
    'train_dataset': processed_dataset,
    'callbacks': callbacks,
}

# Add data_collator only for assistant_only strategy

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
        logger.info(f"[RESUME] Resuming training from checkpoint: {last_checkpoint}")
        send_ntfy(f"Resuming training from checkpoint: {checkpoints[-1]}")

# --- 8. Start Training ---
logger.info("[START] Starting training...")
send_ntfy(f"[START] Starting training job for {config['base_model']}")

# Handle early stopping for sweeps
if sweep_metadata and 'early_stop' in sweep_metadata:
    early_stop = sweep_metadata['early_stop']

    if 'percent_epoch' in early_stop:
        # Calculate number of steps for percentage of epoch
        percent = early_stop['percent_epoch']
        total_steps = len(processed_dataset) // (config.get('batch_size', 1) * config.get('gradient_accumulation', 16))
        max_steps = int(total_steps * percent / 100)
        training_args.max_steps = max_steps
        logger.info(f"Early stopping at {percent}% of epoch ({max_steps} steps)")

    elif 'max_steps' in early_stop:
        # Use explicit step count
        training_args.max_steps = early_stop['max_steps']
        logger.info(f"Early stopping at {early_stop['max_steps']} steps")

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
    history = {
        'loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log],
        'steps': [log['step'] for log in trainer.state.log_history if 'loss' in log],
        'sweep_params': sweep_metadata['sweep_params']
    }
    history_path = config['output_dir'] + '/training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f)

# --- 9. Save and Upload ---
logger.info(f"[SAVE] Saving final model to {config['output_dir']}...")
trainer.save_model(config['output_dir'])
tokenizer.save_pretrained(config['output_dir'])

if config.get('upload_to_hub', False):
    try:
        logger.info(f"📤 Uploading model to Hugging Face Hub: {config['output_model_name']}")
        api = HfApi(token=hf_token if hf_token else None)
        api.create_repo(repo_id=config['output_model_name'], exist_ok=True, private=False)
        api.upload_folder(
            folder_path=config['output_dir'],
            repo_id=config['output_model_name'],
            commit_message="Training run with 'Late' library",
        )
        logger.info(f"[OK] Model successfully uploaded to https://huggingface.co/{config['output_model_name']}")
        send_ntfy(
            f"[SUCCESS] Model uploaded to HF Hub: {config['output_model_name']}",
            title="Upload Complete"
        )
    except Exception as e:
        logger.error(f"[WARN] Error uploading to Hugging Face Hub: {e}")
        send_ntfy(f"[ERROR] Model upload failed: {e}", title="Error")

