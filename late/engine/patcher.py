import subprocess
import sys
import shutil
import os
from pathlib import Path

def run_command(command: str, cwd: str = None, venv_path: str = None):
    """
    Executes a shell command, potentially within a virtual environment,
    and streams its output. Exits the program if the command fails.
    """
    print(f"🔩 Executing: {command}")

    # The command is executed in a shell, which will respect an activated venv
    # if the script itself is run from within one. We don't need complex activation logic.
    process = subprocess.Popen(
        command,
        shell=True,
        executable='/bin/bash', # Use bash for consistency
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    if process.returncode != 0:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"❌ ERROR: Command failed with exit code {process.returncode}", file=sys.stderr)
        print(f"❌ Command: {command}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        sys.exit(process.returncode)

def patch_rocm_environment(arch="gfx942", venv_path: str = None):
    """
    Automates patching for the ROCm environment, targeting either the global
    environment or a specified virtual environment.
    """
    python_executable = sys.executable
    pip_executable = f"{Path(sys.executable).parent}/pip"

    if venv_path:
        venv_path_obj = Path(venv_path).resolve()
        py_exec = venv_path_obj / "bin" / "python"
        pip_exec = venv_path_obj / "bin" / "pip"

        if not py_exec.exists():
            print(f"❌ FATAL ERROR: Python executable not found at '{py_exec}'.", file=sys.stderr)
            print("Please provide a valid path to a virtual environment directory.", file=sys.stderr)
            sys.exit(1)
        
        python_executable = str(py_exec)
        pip_executable = str(pip_exec)
        print(f"🎯 Targeting virtual environment at: {venv_path_obj}")

    else:
        print("🎯 Targeting current/global Python environment.")
        print(f"   - Python: {python_executable}")
        print("   - To target a specific venv, use the --venv flag.")

    print("🚀 Starting ROCm patching process. This will take a while.")

    # --- Prerequisite Check ---
    if shutil.which('git') is None:
        print("❌ FATAL ERROR: 'git' command not found. Please install Git.", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Install Build-time Dependencies ---
    print("\n--- 1. Installing Build-time Dependencies ---")
    run_command(f"{pip_executable} install --upgrade pip")
    run_command(f"{pip_executable} install ninja cmake wheel pybind11")

    # --- Step 2: Build and Install Triton from Source ---
    print("\n--- 2. Building and Installing Triton from Source ---")
    if os.path.exists("triton"): shutil.rmtree("triton")
    run_command("git clone https://github.com/triton-lang/triton.git")
    # Use the targeted pip to install into the correct environment
    run_command(f"{pip_executable} install -e ./python", cwd="triton")
    print("Cleaning up Triton source directory...")
    shutil.rmtree("triton", ignore_errors=True)

    # --- Step 3: Install Core ML Libraries ---
    print("\n--- 3. Installing Core ML Libraries ---")
    run_command(f'{pip_executable} install "numpy<2" "transformers>=4.40.0" "datasets" "accelerate" "trl" "peft" "wandb" "torch" "scipy"')

    # --- Step 4: Build and Install ROCm Flash Attention 2 ---
    print(f"\n--- 4. Building and Installing ROCm Flash Attention 2 for {arch} ---")
    if os.path.exists("flash-attention"): shutil.rmtree("flash-attention")
    run_command("git clone https://github.com/ROCm/flash-attention.git")
    # Use the targeted python executable to run the setup script
    build_command = f"MAX_JOBS=$(nproc) GPU_ARCHS={arch} {python_executable} setup.py install"
    run_command(build_command, cwd="flash-attention")
    print("Cleaning up Flash Attention source directory...")
    shutil.rmtree("flash-attention", ignore_errors=True)

    # --- Step 5: Verification ---
    print("\n--- 5. Verifying Installation in Target Environment ---")
    verify_script = (
        "import importlib; "
        "importlib.invalidate_caches(); "
        "import triton; "
        "import flash_attn; "
        "print(f'Successfully imported Triton version: {triton.__version__}'); "
        "print(f'Successfully imported flash_attn version: {flash_attn.__version__}')"
    )
    
    try:
        # Execute the verification script using the target python
        print("Verifying packages...")
        run_command(f"{python_executable} -c '{verify_script}'")
        print("\n🎉 Patching complete. Your environment is ready.")
    except SystemExit:
        print(f"\n❌ VERIFICATION FAILED. Could not import libraries in the target environment.", file=sys.stderr)
        print("Please check the build logs above for errors.", file=sys.stderr)
        sys.exit(1)