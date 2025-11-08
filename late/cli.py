import click
import os
from late.engine.queue_manager import QueueManager
from late.engine.config import save_token

# Lazy imports for optional dependencies (only imported when needed)
# This allows the CLI and dashboard to work without torch/training deps installed

qm = QueueManager()

PURPLE = '\033[95m'
RESET = '\033[0m'
GRAY = '\033[90m'

BANNER = f"""{PURPLE}
██╗      █████╗ ████████╗███████╗
██║     ██╔══██╗╚══██╔══╝██╔════╝
██║     ███████║   ██║   █████╗
██║     ██╔══██║   ██║   ██╔══╝
███████╗██║  ██║   ██║   ███████╗
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
{RESET}{GRAY}A powerful toolkit for streamlining and scheduling ML training workflows.{RESET}
"""

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Late: A toolkit for patching, scheduling, and running training jobs on ROCm."""
    if ctx.invoked_subcommand is None:
        click.echo(BANNER)
        click.echo(ctx.get_help())
    else:
        click.echo(BANNER)

@cli.command()
def version():
    """Show the version of Late."""
    click.echo(BANNER)
    click.echo("Late version 0.1.0")

@cli.command()
def clear():
    """Clear all GPU VRAM and training-related memory.

    This command clears:
    - GPU VRAM memory (CUDA/ROCm)
    - Python garbage collection
    - PyTorch caches

    Useful before starting new training runs to ensure clean state.
    """
    from late.engine.training import clear_memory
    import torch

    click.echo("🧹 Clearing all training-related memory...")

    # Detect platform
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name('cuda')
        if hasattr(torch.version, 'hip') and torch.version.hip and 'rocm' in torch.version.hip:
            platform = f"AMD ROCm ({device_name})"
        else:
            platform = f"NVIDIA CUDA ({device_name})"

        # Get memory before clearing
        allocated_before = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved_before = torch.cuda.memory_reserved() / 1024**3    # GB

        click.echo(f"📊 Platform: {platform}")
        click.echo(f"📊 VRAM Allocated: {allocated_before:.2f} GB")
        click.echo(f"📊 VRAM Reserved: {reserved_before:.2f} GB")
        click.echo("")
    else:
        platform = "CPU"
        click.echo(f"📊 Platform: {platform}")
        click.echo("")

    # Clear memory
    clear_memory()

    # Show memory after clearing
    if torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated() / 1024**3   # GB
        reserved_after = torch.cuda.memory_reserved() / 1024**3     # GB
        freed_allocated = allocated_before - allocated_after
        freed_reserved = reserved_before - reserved_after

        click.echo(f"✅ Memory cleared successfully!")
        click.echo(f"📉 VRAM Allocated: {allocated_after:.2f} GB (freed {freed_allocated:.2f} GB)")
        click.echo(f"📉 VRAM Reserved: {reserved_after:.2f} GB (freed {freed_reserved:.2f} GB)")
    else:
        click.echo(f"✅ Memory cleared successfully!")

    click.echo("")
    click.echo("Ready for training! 🚀")

@cli.group()
def patch():
    """Patch PyTorch environment for optimal performance (amd, nvidia, or cpu)."""
    pass

@patch.command('amd')
@click.option('--arch', default='gfx942', help='Target GPU architecture(s), semicolon-separated (e.g., "gfx942;gfx90a").')
@click.option('--venv', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              help='Path to the virtual environment directory to patch.')
@click.option('--rocm-version', default='latest', help="ROCm version for MIOpen kernels (e.g., '6.1.3').")
@click.option('--no-kernels', is_flag=True, default=False, help='Skip installing MIOpen kernel databases.')
@click.option('--from-source', is_flag=True, default=False, help='Build Flash Attention from source instead of using pre-built wheel.')
@click.option('--install-pytorch', default=None, help='Install PyTorch: "stable", "nightly", or a wheel URL. Default: skip PyTorch installation.')
def patch_amd(arch, venv, rocm_version, no_kernels, from_source, install_pytorch):
    """Patch environment with AMD ROCm optimizations."""
    from late.engine.patcher import patch_rocm_environment

    click.echo("--- Starting AMD ROCm Patching Process ---")
    click.echo(f"  - Target Architecture(s): {arch}")
    click.echo(f"  - Target Environment: {'Virtual env at ' + os.path.abspath(venv) if venv else 'Current active or global environment'}")
    if install_pytorch:
        if install_pytorch == "stable":
            pytorch_msg = "Installing STABLE PyTorch for ROCm"
        elif install_pytorch == "nightly":
            pytorch_msg = "Installing NIGHTLY PyTorch for ROCm 6.4"
        else:
            pytorch_msg = "Installing PyTorch from provided URL"
    else:
        pytorch_msg = "Skipping (use --install-pytorch to install)"

    click.echo(f"  - PyTorch: {pytorch_msg}")
    click.echo(f"  - MIOpen Kernels: {'Skipping' if no_kernels else 'Installing for ROCm ' + rocm_version}")
    click.echo(f"  - Flash Attention: {'Building from source' if from_source else 'Using pre-built wheel'}")

    patch_rocm_environment(
        arch=arch,
        venv_path=venv,
        rocm_version=rocm_version,
        install_kernels=not no_kernels,
        build_from_source=from_source,
        pytorch_install=install_pytorch
    )

@patch.command('nvidia')
@click.option('--venv', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              help='Path to the virtual environment directory to patch.')
@click.option('--install-pytorch', default=None, help='Install PyTorch: "stable", "nightly", or a wheel URL. Default: skip PyTorch installation.')
def patch_nvidia(venv, install_pytorch):
    """Patch environment with NVIDIA CUDA optimizations."""
    import subprocess
    click.echo("--- Starting NVIDIA CUDA Patching Process ---")
    click.echo(f"  - Target Environment: {'Virtual env at ' + os.path.abspath(venv) if venv else 'Current active or global environment'}")

    # Install PyTorch with CUDA support
    if install_pytorch:
        click.echo("[INFO] Installing PyTorch with CUDA support...")
        if install_pytorch == "stable":
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif install_pytorch == "nightly":
            cmd = "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121"
        else:
            cmd = f"pip install {install_pytorch}"

        subprocess.run(cmd, shell=True, check=True)
        click.echo("[OK] PyTorch installed")

    # Install Flash Attention for NVIDIA
    click.echo("[INFO] Installing Flash Attention 2 for NVIDIA...")
    try:
        subprocess.run("pip install flash-attn --no-build-isolation", shell=True, check=True)
        click.echo("[OK] Flash Attention 2 installed")
    except:
        click.echo("[WARN] Flash Attention 2 installation failed. Continuing without it.")

    click.echo("[SUCCESS] NVIDIA CUDA environment patched successfully!")

@patch.command('cpu')
@click.option('--venv', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              help='Path to the virtual environment directory to patch.')
@click.option('--install-pytorch', default=None, help='Install PyTorch: "stable", "nightly", or a wheel URL. Default: skip PyTorch installation.')
def patch_cpu(venv, install_pytorch):
    """Patch environment for CPU-only training."""
    import subprocess
    click.echo("--- Starting CPU-Only Patching Process ---")
    click.echo(f"  - Target Environment: {'Virtual env at ' + os.path.abspath(venv) if venv else 'Current active or global environment'}")

    # Install CPU-only PyTorch
    if install_pytorch:
        click.echo("[INFO] Installing CPU-only PyTorch...")
        if install_pytorch == "stable":
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        elif install_pytorch == "nightly":
            cmd = "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu"
        else:
            cmd = f"pip install {install_pytorch}"

        subprocess.run(cmd, shell=True, check=True)
        click.echo("[OK] CPU-only PyTorch installed")

    click.echo("[SUCCESS] CPU environment patched successfully!")
    click.echo("[INFO] Note: CPU training will be significantly slower than GPU training.")

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Run a single training job from a YAML config file."""
    from late.engine.training import run_training_job
    run_training_job(config_path)

@cli.command()
@click.argument('adapter_path', type=click.Path(exists=True))
@click.option('--base-model', required=True, help='Base model name from HuggingFace (e.g., "Qwen/Qwen3-32B")')
@click.option('--output', required=True, help='Output repository ID (e.g., "username/merged-model")')
@click.option('--local-path', help='Local directory to save merged model (default: /tmp/late_merged_model)')
@click.option('--no-upload', is_flag=True, help='Skip uploading to HuggingFace Hub')
@click.option('--private', is_flag=True, help='Make the Hub repository private')
@click.option('--config', type=click.Path(exists=True), help='Use YAML config file instead of CLI options')
def merge(adapter_path, base_model, output, local_path, no_upload, private, config):
    """Merge a LoRA adapter with its base model and upload to HuggingFace Hub.

    Examples:
        # Merge and upload to Hub
        late merge /path/to/checkpoint-100 --base-model Qwen/Qwen3-32B --output username/merged-model

        # Merge only (no upload)
        late merge /path/to/checkpoint-100 --base-model Qwen/Qwen3-32B --output username/merged-model --no-upload

        # Use config file
        late merge /path/to/checkpoint-100 --config merge_config.yml
    """
    from late.engine.merge_lora import merge_lora_adapter, merge_lora_from_config

    if config:
        # Use config file mode
        click.echo(f"Using configuration from: {config}")
        result = merge_lora_from_config(config)
    else:
        # Use CLI arguments
        if not base_model or not output:
            click.echo("Error: --base-model and --output are required when not using --config")
            return

        click.echo(f"Merging LoRA adapter...")
        click.echo(f"   Adapter: {adapter_path}")
        click.echo(f"   Base Model: {base_model}")
        click.echo(f"   Output: {output}")

        result = merge_lora_adapter(
            adapter_path=adapter_path,
            base_model_name=base_model,
            output_repo_id=output,
            local_save_path=local_path,
            upload_to_hub=not no_upload,
            private_repo=private
        )

    click.echo(f"\n{'='*80}")
    click.echo(f"Merge complete!")
    click.echo(f"Result: {result}")
    click.echo(f"{'='*80}\n")

@cli.command()
@click.argument('base_config', type=click.Path(exists=True))
@click.option('--sweep-file', type=click.Path(exists=True), help='Path to .sweep file with parameters')
@click.option('--params', multiple=True, help='Sweep parameters (e.g., learning_rate=1e-4,2e-4,3e-4)')
@click.option('--override', multiple=True, help='Override parameters (e.g., override.max_seq_length=2048)')
@click.option('--sweep-id', help='Custom sweep identifier for W&B grouping')
@click.option('--percent-epoch', type=float, help='Stop each run at X% of one epoch')
@click.option('--max-steps', type=int, help='Stop each run at X steps')
@click.option('--add-to-queue', help='Add sweep configs to specified queue instead of running')
def sweep(base_config, sweep_file, params, override, sweep_id, percent_epoch, max_steps, add_to_queue):
    from late.engine.sweep import create_sweep_configs, save_sweep_configs
    from late.engine.sweep_report import generate_sweep_report
    from late.engine.training import run_training_job
    """Run hyperparameter sweeps for LoRA training.
    
    Examples:
        # Sweep learning rate with 25% epoch early stopping
        late sweep config.yml --params learning_rate=1e-4,2e-4,3e-4 --percent-epoch 25
        
        # Sweep multiple parameters with shorter context
        late sweep config.yml --params learning_rate=1e-4,2e-4 lora.r=64,128 --override max_seq_length=2048
        
        # Use a sweep file
        late sweep config.yml --sweep-file my_sweep.sweep
        
        # Add to queue instead of running immediately
        late sweep config.yml --sweep-file my_sweep.sweep --add-to-queue overnight.qml
    """
    from pathlib import Path
    
    # Build sweep config
    sweep_config = {'sweep_parameters': {}, 'overrides': {}}
    
    # Add sweep ID if provided
    if sweep_id:
        sweep_config['sweep_id'] = sweep_id
    
    # Add early stopping config
    if percent_epoch or max_steps:
        sweep_config['early_stop'] = {}
        if percent_epoch:
            sweep_config['early_stop']['percent_epoch'] = percent_epoch
        if max_steps:
            sweep_config['early_stop']['max_steps'] = max_steps
    
    # Parse parameters and overrides
    all_params = []
    if params:
        all_params.extend(params)
    if override:
        all_params.extend([f"override.{o}" for o in override])
    
    # Create sweep configs
    if sweep_file:
        configs = create_sweep_configs(base_config, sweep_file)
    elif all_params:
        configs = create_sweep_configs(base_config, all_params)
        # Merge in the sweep_config settings
        for config in configs:
            if sweep_id:
                config['_sweep_metadata']['sweep_id'] = sweep_id
            if 'early_stop' in sweep_config:
                config['_sweep_metadata']['early_stop'] = sweep_config['early_stop']
    else:
        click.echo("Error: Must provide either --sweep-file or --params")
        return
    
    # Save sweep configs
    config_paths, sweep_dir = save_sweep_configs(configs)
    
    click.echo(f"📋 Generated {len(config_paths)} sweep configurations")
    click.echo(f"📁 Saved to: {sweep_dir}")
    
    if add_to_queue:
        # Add all configs to queue
        for config_path in config_paths:
            qm.add_to_queue(add_to_queue, config_path)
        click.echo(f"✅ Added {len(config_paths)} configs to queue '{add_to_queue}'")
    else:
        # Run immediately
        click.echo("\n🚀 Starting sweep runs...")
        for i, config_path in enumerate(config_paths):
            click.echo(f"\n--- Running sweep {i+1}/{len(config_paths)} ---")
            run_training_job(config_path)
            
            # Clear memory between runs
            from late.engine.training import clear_memory
            clear_memory()
        
        # Generate report
        click.echo("\n📊 Generating sweep report...")
        try:
            report_path = generate_sweep_report(Path(sweep_dir))
            click.echo(f"✅ Sweep report saved to: {report_path}")
        except Exception as e:
            click.echo(f"⚠️ Failed to generate report: {e}")
        
        click.echo(f"\n🎉 Sweep complete! Results in: {sweep_dir}")

@cli.group()
def queue():
    """Manage training queues (.qml files)."""
    pass

@queue.command('ls')
def list_queues():
    """List all available training queues."""
    queues = qm.list_queues()
    if not queues:
        click.echo("No queues found. Create one with `late queue create <name>.qml`")
        return
    click.echo("Available Queues:")
    for q in queues:
        click.echo(f"- {q}")

@queue.command('create')
@click.argument('queue_name')
def create_queue(queue_name):
    """Create a new, empty training queue."""
    qm.create_queue(queue_name)

@queue.command('add')
@click.argument('queue_name')
@click.argument('config_path', type=click.Path(exists=True))
def add_to_queue(queue_name, config_path):
    """Add a training config to a queue."""
    qm.add_to_queue(queue_name, config_path)

@queue.command('delete')
@click.argument('queue_name')
def delete_queue(queue_name):
    """Delete a training queue."""
    qm.delete_queue(queue_name)

@queue.command('start')
@click.argument('queue_name')
@click.option('--resume/--restart', default=True, help='Resume from last completed job or restart from beginning')
def start_queue(queue_name, resume):
    """Start running all training jobs in a queue sequentially."""
    from late.engine.training import clear_memory
    
    jobs = qm.get_queue_contents(queue_name)
    if not jobs:
        click.echo(f"Queue '{queue_name}' is empty or does not exist.")
        return
    
    # Check for existing progress
    start_index = 0
    if resume:
        last_completed = qm.get_progress(queue_name)
        if last_completed >= 0:
            start_index = last_completed + 1
            click.echo(f"📌 Resuming queue '{queue_name}' from job {start_index + 1}/{len(jobs)}")
        else:
            click.echo(f"Starting queue '{queue_name}' with {len(jobs)} jobs.")
    else:
        qm.clear_progress(queue_name)
        click.echo(f"Starting queue '{queue_name}' with {len(jobs)} jobs (fresh start).")
    
    # Check if this is a sweep queue
    is_sweep_queue = False
    sweep_dirs = set()
    for job in jobs[start_index:]:
        if "sweep_runs" in job:
            is_sweep_queue = True
            # Extract sweep directory
            parts = Path(job).parts
            if "sweep_runs" in parts:
                sweep_idx = parts.index("sweep_runs")
                if sweep_idx + 1 < len(parts):
                    sweep_dir = Path(*parts[:sweep_idx+2])
                    sweep_dirs.add(sweep_dir)
    
    try:
        for i in range(start_index, len(jobs)):
            config_path = jobs[i]
            click.echo(f"\n--- Running job {i+1}/{len(jobs)} ---")
            if not os.path.exists(config_path):
                click.echo(f"[WARN] Config file not found, skipping: {config_path}", err=True)
                qm.save_progress(queue_name, i)
                continue
            
            # Clear memory before each run
            click.echo("Clearing memory before run...")
            clear_memory()
            
            run_training_job(config_path)
            
            # Save progress after successful completion
            qm.save_progress(queue_name, i)
            
            # Clear memory after each run
            click.echo("Clearing memory after run...")
            clear_memory()
            
            # Small delay between runs to ensure cleanup
            import time
            time.sleep(2)
        
        click.echo(f"\nQueue '{queue_name}' finished.")
        qm.clear_progress(queue_name)  # Clear progress on successful completion
        
        # Generate sweep reports if this was a sweep queue
        if is_sweep_queue and sweep_dirs:
            click.echo("\n📊 Generating sweep reports...")
            for sweep_dir in sweep_dirs:
                try:
                    from late.engine.sweep_report import generate_sweep_report
                    report_path = generate_sweep_report(sweep_dir)
                    click.echo(f"✅ Sweep report saved to: {report_path}")
                except Exception as e:
                    click.echo(f"⚠️ Failed to generate report for {sweep_dir}: {e}")
                    
    except KeyboardInterrupt:
        click.echo(f"\n\n⚠️ Queue interrupted. Progress saved at job {i+1}/{len(jobs)}")
        click.echo(f"Run 'late queue start {queue_name}' to resume.")
        return
    except Exception as e:
        click.echo(f"\n❌ Queue failed with error: {e}")
        click.echo(f"Progress saved. Run 'late queue start {queue_name}' to resume from job {i+1}.")
        raise

@cli.command()
@click.argument('key', type=click.Choice(['wandb', 'hf_token'], case_sensitive=False))
@click.argument('token')
def set(key, token):
    """Set global API tokens (wandb, hf_token)."""
    save_token(key, token)

@cli.command()
@click.option('--port', default=8080, help='Port to run the web dashboard on.')
def serve(port):
    """Launch the web dashboard."""
    from late.server.app import run_server

    click.echo(f"Launching web dashboard at http://0.0.0.0:{port}")
    click.echo("Access it from your browser. Use Ctrl+C to shut down.")
    run_server(port=port)

@cli.command('start')
@click.argument('queue_name')
@click.option('--resume/--restart', default=True, help='Resume from last completed job or restart from beginning')
def start(queue_name, resume):
    """Start running a training queue (alias for 'queue start')."""
    from late.engine.training import clear_memory

    jobs = qm.get_queue_contents(queue_name)
    if not jobs:
        click.echo(f"Queue '{queue_name}' is empty or does not exist.")
        return

    # Check for existing progress
    start_index = 0
    if resume:
        last_completed = qm.get_progress(queue_name)
        if last_completed >= 0:
            start_index = last_completed + 1
            click.echo(f"[INFO] Resuming queue '{queue_name}' from job {start_index + 1}/{len(jobs)}")
        else:
            click.echo(f"Starting queue '{queue_name}' with {len(jobs)} jobs.")
    else:
        qm.clear_progress(queue_name)
        click.echo(f"Starting queue '{queue_name}' with {len(jobs)} jobs (fresh start).")

    # Check if this is a sweep queue
    is_sweep_queue = False
    sweep_dirs = set()
    for job in jobs[start_index:]:
        if "sweep_runs" in job:
            is_sweep_queue = True
            # Extract sweep directory
            parts = Path(job).parts
            if "sweep_runs" in parts:
                sweep_idx = parts.index("sweep_runs")
                if sweep_idx + 1 < len(parts):
                    sweep_dir = Path(*parts[:sweep_idx+2])
                    sweep_dirs.add(sweep_dir)

    try:
        for i in range(start_index, len(jobs)):
            config_path = jobs[i]
            click.echo(f"\n--- Running job {i+1}/{len(jobs)} ---")
            if not os.path.exists(config_path):
                click.echo(f"[WARN] Config file not found, skipping: {config_path}", err=True)
                qm.save_progress(queue_name, i)
                continue

            # Clear memory before each run
            click.echo("Clearing memory before run...")
            clear_memory()

            from late.engine.training import run_training_job
            run_training_job(config_path)

            # Save progress after successful completion
            qm.save_progress(queue_name, i)

            # Clear memory after each run
            click.echo("Clearing memory after run...")
            clear_memory()

            # Small delay between runs to ensure cleanup
            import time
            time.sleep(2)

        click.echo(f"\nQueue '{queue_name}' finished.")
        qm.clear_progress(queue_name)  # Clear progress on successful completion

        # Generate sweep reports if this was a sweep queue
        if is_sweep_queue and sweep_dirs:
            click.echo("\n[INFO] Generating sweep reports...")
            for sweep_dir in sweep_dirs:
                try:
                    from late.engine.sweep_report import generate_sweep_report
                    report_path = generate_sweep_report(sweep_dir)
                    click.echo(f"[OK] Sweep report saved to: {report_path}")
                except Exception as e:
                    click.echo(f"[WARN] Failed to generate report for {sweep_dir}: {e}")

    except KeyboardInterrupt:
        click.echo(f"\n\n[WARN] Queue interrupted. Progress saved at job {i+1}/{len(jobs)}")
        click.echo(f"Run 'late start {queue_name}' to resume.")
        return
    except Exception as e:
        click.echo(f"\n[ERROR] Queue failed with error: {e}")
        click.echo(f"Progress saved. Run 'late start {queue_name}' to resume from job {i+1}.")
        raise

if __name__ == '__main__':
    cli()
