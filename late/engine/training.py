import subprocess
import sys
import yaml
from .config import load_tokens

def run_training_job(config_path: str):
    """
    Runs a training job by generating a Python script from a YAML config
    and executing it.
    """
    print(f"\n🚀 Launching training job for: {config_path}\n{'='*50}")

    # Load tokens into environment for the subprocess
    load_tokens()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    script_content = generate_training_script(config)
    
    script_path = "temp_training_script.py"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Execute the generated script
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        print(f"\n{'='*50}\n✅ Training job for '{config_path}' completed successfully.")
    else:
        print(f"\n{'='*50}\n❌ Training job for '{config_path}' failed with exit code {process.returncode}.")

def generate_training_script(config: dict) -> str:
    """Generates the full Python training script based on the config."""
    
    # This function is large because it contains the full, dynamic script.
    # It directly translates the user-provided notebook cells into a runnable script.
    
    # Determine training type (SFT or LoRA)
    is_lora = config.get('training_type', '').lower() == 'lora'

    # Build the script string
    script = f"""
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if not torch.cuda.is_available() or 'rocm' not in torch.version.hip:
    raise RuntimeError("ROCm GPU not found. This script requires a ROCm environment.")
logger.info(f"✅ Detected ROCm-enabled GPU: {{torch.cuda.get_device_name(0)}}")

config = {config}

# --- 2. W&B Setup ---
if config.get('report_to_wandb', False):
    os.environ.setdefault("WANDB_PROJECT", "Late-Training-Runs")
    logger.info("W&B reporting is enabled.")

# --- 3. Model and Tokenizer Loading ---
logger.info(f"Loading base model: {{config['base_model']}}")
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    torch_dtype=dtype,
    attn_implementation="flash_attention_2",
    use_cache=False,
    device_map={{'': torch.cuda.current_device()}},
    cache_dir=config.get('cache_dir', './model_cache'),
)
tokenizer = AutoTokenizer.from_pretrained(config['base_model'], cache_dir=config.get('cache_dir', './model_cache'))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. LoRA Configuration (if applicable) ---
"""
    if is_lora:
        script += f"""
logger.info("Applying LoRA configuration...")
lora_config_data = config.get('lora', {{}})
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
    
    script += f"""
# --- 5. Dataset Loading and Preprocessing ---
logger.info(f"Loading dataset: {{config['dataset_name']}}")
dataset = load_dataset(config['dataset_name'], split="train")
tokenizer.model_max_length = config['max_seq_length']

def preprocess_for_assistant_loss(examples, tokenizer):
    all_input_ids, all_labels = [], []
    for messages in examples["messages"]:
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

logger.info("Applying preprocessing to mask user prompts from loss calculation...")
processed_dataset = dataset.map(
    preprocess_for_assistant_loss,
    fn_kwargs={{"tokenizer": tokenizer}},
    batched=True,
    batch_size=100,
    remove_columns=dataset.column_names,
)

# --- 6. Trainer Configuration ---
logger.info("Configuring SFTTrainer...")
training_args = SFTConfig(
    output_dir=config['output_dir'],
    num_train_epochs=config.get('epochs', 1),
    per_device_train_batch_size=config.get('batch_size', 1),
    gradient_accumulation_steps=config.get('gradient_accumulation', 16),
    learning_rate=config.get('learning_rate', 2e-5),
    lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
    optim=config.get('optim', 'adamw_torch_fused'),
    max_seq_length=config['max_seq_length'],
    bf16=True,
    torch_compile=True,
    logging_steps=10,
    save_steps=50,
    save_strategy="steps",
    save_total_limit=5,
    report_to=["wandb"] if config.get('report_to_wandb') else "none",
    run_name=f"{{config['output_model_name'].split('/')[-1]}}-{{config.get('training_type', 'sft')}}",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

# --- 7. Start Training ---
logger.info("🚀 Starting training...")
trainer.train()
logger.info("✅ Training complete.")

# --- 8. Save and Upload ---
logger.info(f"💾 Saving final model to {{config['output_dir']}}...")
trainer.save_model(config['output_dir'])
tokenizer.save_pretrained(config['output_dir'])

if config.get('upload_to_hub', False):
    try:
        logger.info(f"📤 Uploading model to Hugging Face Hub: {{config['output_model_name']}}")
        api = HfApi()
        api.create_repo(repo_id=config['output_model_name'], exist_ok=True, private=False)
        api.upload_folder(
            folder_path=config['output_dir'],
            repo_id=config['output_model_name'],
            commit_message="Training run with 'Late' library",
        )
        logger.info(f"✓ Model successfully uploaded to https://huggingface.co/{{config['output_model_name']}}")
    except Exception as e:
        logger.error(f"⚠️ Error uploading to Hugging Face Hub: {{e}}")

"""
    return script