"""
LoRA Adapter Merging Module

This module provides functionality to merge trained LoRA adapters with their base models
and optionally upload the merged model to the Hugging Face Hub.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def merge_lora_adapter(
    adapter_path: str,
    base_model_name: str,
    output_repo_id: str,
    local_save_path: Optional[str] = None,
    upload_to_hub: bool = True,
    private_repo: bool = False,
    hf_token: Optional[str] = None
) -> str:
    """
    Merges a LoRA adapter with its base model and optionally uploads to Hugging Face Hub.

    Args:
        adapter_path: Local path to the trained LoRA adapter directory
        base_model_name: HuggingFace model ID for the base model
        output_repo_id: Repository ID for the merged model (format: "username/model-name")
        local_save_path: Optional local directory to save the merged model
        upload_to_hub: Whether to upload to HuggingFace Hub (default: True)
        private_repo: Whether the Hub repository should be private (default: False)
        hf_token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)

    Returns:
        Path to the merged model (local path or Hub URL)

    Raises:
        FileNotFoundError: If adapter_path doesn't exist
        ValueError: If configuration is invalid
    """

    # Validate inputs
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: '{adapter_path}'. Please check the path.")

    if not local_save_path:
        local_save_path = "/tmp/late_merged_model"

    local_save_path = Path(local_save_path)

    # Set up HuggingFace token
    if hf_token:
        HfFolder.save_token(hf_token)
        logger.info("HuggingFace token is set.")
    elif os.environ.get('HF_TOKEN'):
        HfFolder.save_token(os.environ['HF_TOKEN'])
        logger.info("Using HF_TOKEN from environment.")

    logger.info(f"[START] Starting LoRA merge process...")
    logger.info(f"   Base model: {base_model_name}")
    logger.info(f"   Adapter path: {adapter_path}")
    logger.info(f"   Output repo: {output_repo_id}")

    # --- Load Base Model ---
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        return_dict=True,
    )
    logger.info("[OK] Base model loaded.")

    # --- Load and Apply LoRA Adapter ---
    logger.info(f"Applying LoRA adapter from: {adapter_path}")
    # The PeftModel class will automatically load the adapter and apply it
    model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("[OK] LoRA adapter applied.")

    # --- Merge the Adapter into the Base Model ---
    logger.info("Merging the LoRA adapter into the base model...")
    # This is the key step that combines the weights. `merge_and_unload` returns the
    # standard transformers model with the merged weights.
    model = model.merge_and_unload()
    logger.info("[OK] LoRA adapter successfully merged and unloaded.")

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer for {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    logger.info("[OK] Tokenizer loaded.")

    # --- Save Merged Model Locally ---
    logger.info(f"[SAVE] Saving merged model and tokenizer to local path: {local_save_path}")
    local_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_save_path)
    tokenizer.save_pretrained(local_save_path)
    logger.info("[OK] Merged model saved locally.")

    # --- Upload to Hugging Face Hub ---
    if upload_to_hub:
        try:
            api = HfApi()

            logger.info(f"[UPLOAD] Creating repository '{output_repo_id}' on the Hub...")
            api.create_repo(
                repo_id=output_repo_id,
                exist_ok=True,
                repo_type="model",
                private=private_repo
            )
            logger.info(f"[OK] Repository created (private={private_repo}).")

            logger.info(f"Uploading all model files from '{local_save_path}'...")
            api.upload_folder(
                folder_path=str(local_save_path),
                repo_id=output_repo_id,
                commit_message="feat: Upload merged LoRA model via Late",
            )

            hub_url = f"https://huggingface.co/{output_repo_id}"
            logger.info(f"[SUCCESS] Successfully uploaded merged model to: {hub_url}")
            return hub_url

        except Exception as e:
            logger.error(f"[WARN] Error uploading to Hugging Face Hub: {e}")
            logger.info(f"Model is still saved locally at: {local_save_path}")
            return str(local_save_path)
    else:
        logger.info(f"[OK] Merge complete. Model saved locally at: {local_save_path}")
        return str(local_save_path)

def merge_lora_from_config(config_path: str) -> str:
    """
    Merges a LoRA adapter using a YAML configuration file.

    Args:
        config_path: Path to YAML config file with merge settings

    Returns:
        Path to merged model (local or Hub URL)

    Example config:
        ```yaml
        adapter_path: "/path/to/checkpoint-100"
        base_model: "Qwen/Qwen3-32B"
        output_repo_id: "username/merged-model"
        local_save_path: "./merged_output"
        upload_to_hub: true
        private_repo: false
        ```
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    required_fields = ['adapter_path', 'base_model', 'output_repo_id']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    return merge_lora_adapter(
        adapter_path=config['adapter_path'],
        base_model_name=config['base_model'],
        output_repo_id=config['output_repo_id'],
        local_save_path=config.get('local_save_path'),
        upload_to_hub=config.get('upload_to_hub', True),
        private_repo=config.get('private_repo', False),
        hf_token=config.get('hf_token')
    )

if __name__ == "__main__":
    """Example usage when run as a script."""
    import sys

    if len(sys.argv) < 4:
        print("Usage: python merge_lora.py <adapter_path> <base_model> <output_repo_id> [local_save_path]")
        sys.exit(1)

    adapter_path = sys.argv[1]
    base_model = sys.argv[2]
    output_repo_id = sys.argv[3]
    local_save_path = sys.argv[4] if len(sys.argv) > 4 else None

    result = merge_lora_adapter(
        adapter_path=adapter_path,
        base_model_name=base_model,
        output_repo_id=output_repo_id,
        local_save_path=local_save_path
    )

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Merge complete!")
    print(f"[INFO] Result: {result}")
    print(f"{'='*80}\n")
