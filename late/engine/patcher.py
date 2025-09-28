import subprocess
import sys

def run_command(command: str):
    """Executes a shell command and streams its output."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"❌ ERROR: Command failed with exit code {process.returncode}", file=sys.stderr)
        sys.exit(process.returncode)

def patch_rocm_environment(arch="gfx942"):
    """
    Automates the patching process for the ROCm environment, focusing on a specific GPU architecture.
    """
    print("🚀 Starting ROCm environment patching process...")

    print("\n--- 1. Installing Prerequisite System Packages ---")
    run_command("python3 -m pip install ninja cmake wheel pybind11")

    print("\n--- 2. Re-installing Triton from Source ---")
    run_command("pip uninstall -y triton")
    run_command("git clone https://github.com/OpenAI/triton.git && cd triton && git checkout e192dba && cd python && pip3 install . && cd ../.. && rm -rf triton")
    
    print("\n--- 3. Installing Standard Python ML Packages ---")
    run_command('pip install -q "transformers>=4.40.0" "datasets" "accelerate" "trl" "peft" "wandb"')

    print(f"\n--- 4. Installing ROCm Flash Attention 2 from Source for {arch} ---")
    run_command("git clone https://github.com/ROCm/flash-attention.git")
    
    # Use MAX_JOBS to speed up the build
    build_command = f"cd flash-attention && MAX_JOBS=$(nproc) GPU_ARCHS={arch} python setup.py install && cd .."
    run_command(build_command)
    run_command("rm -rf flash-attention") # Clean up

    print("\n--- 5. Verifying Installation ---")
    try:
        import importlib
        importlib.invalidate_caches()
        import flash_attn
        print(f"✅ Successfully imported flash_attn version: {flash_attn.__version__}")
        print("Patching complete. Your environment is ready.")
    except ImportError as e:
        print(f"❌ ERROR: Failed to import flash_attn after installation: {e}")
        print("A manual check might be required.")