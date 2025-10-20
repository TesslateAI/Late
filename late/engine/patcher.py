import subprocess
import sys
import shutil
import os
import re
import glob
import requests
import json
from pathlib import Path
from typing import Optional, Tuple

def run_command(command: str, cwd: str = None):
    """Executes a shell command and streams its output."""
    print(f"[EXEC] {command}")
    process = subprocess.Popen(
        command, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"\n{'='*60}\n[ERROR] Command failed with exit code {process.returncode}\n[ERROR] Command: {command}\n{'='*60}", file=sys.stderr)
        sys.exit(process.returncode)

def _get_pytorch_path(pip_executable: str) -> Path:
    """Finds the installation path of PyTorch in the target environment."""
    try:
        print("[INFO] Finding PyTorch installation path...")
        result = subprocess.check_output([pip_executable, 'show', 'torch'], text=True)
        for line in result.splitlines():
            if line.startswith('Location:'):
                path = Path(line.split(':', 1)[1].strip())
                print(f"[OK] Found PyTorch at: {path}")
                return path
        raise FileNotFoundError("Could not parse 'Location:' from pip output.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] PyTorch is not installed or could not be found in the target environment.", file=sys.stderr)
        print("   'late patch' must install torch before this step. Check the dependency file.", file=sys.stderr)
        sys.exit(1)

def _install_miopen_kernels(torch_path: Path, rocm_version: str, gfx_archs: list[str]):
    """Downloads and installs the MIOpen kernel databases."""
    print("\n--- Installing MIOpen Kernel Databases for PyTorch ---")
    
    distro, version_id = "", ""
    if Path("/etc/lsb-release").exists():
        with open("/etc/lsb-release", "r") as f:
            for line in f:
                if line.startswith("DISTRIB_RELEASE="):
                    distro = "ubuntu"
                    version_id = line.split("=")[1].strip()
    elif Path("/etc/os-release").exists():
        with open("/etc/os-release", "r") as f:
            lines = {k.strip(): v.strip().strip('"') for k, v in (l.split("=", 1) for l in f if "=" in l)}
            if "ID" in lines and lines["ID"] in ["centos", "rhel"]:
                distro = "rhel"
                version_id = lines.get("VERSION_ID", "0").split('.')[0]

    if not distro:
        print("[WARN] Unsupported OS for MIOpen kernel download. Skipping.", file=sys.stderr)
        return

    print(f"[INFO] OS Detected: {distro.capitalize()} {version_id}")
    
    with requests.Session() as s:
        temp_dir = Path("./miopen_kernels_temp")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        try:
            for arch in gfx_archs:
                print(f"  -> Fetching kernels for {arch}...")
                
                if distro == "ubuntu":
                    repo_url = f"https://repo.radeon.com/rocm/apt/{rocm_version}/pool/main/m/"
                    pkg_pattern = re.compile(rf'href="(miopen-hip-{arch}[^"]*kdb_[^"]*{version_id}[^"]*\.deb)"')
                    pkg_type = "deb"
                else: # rhel
                    repo_url = f"https://repo.radeon.com/rocm/rhel{version_id}/{rocm_version}/main/"
                    pkg_pattern = re.compile(rf'href="(miopen-hip-{arch}[^"]*kdb-[^"]*\.rpm)"')
                    pkg_type = "rpm"
                
                response = s.get(repo_url)
                response.raise_for_status()
                matches = pkg_pattern.findall(response.text)
                if not matches:
                    print(f"    [WARN] No MIOpen kernel package found for {arch} at {repo_url}. Skipping.", file=sys.stderr)
                    continue

                pkg_filename = matches[0]
                download_url = f"{repo_url}{pkg_filename}"
                local_path = temp_dir / pkg_filename
                
                print(f"    Downloading {pkg_filename}...")
                with s.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

                print(f"    Extracting {pkg_filename}...")
                if pkg_type == "deb":
                    run_command(f"dpkg-deb -xv {local_path.name} .", cwd=temp_dir)
                else:
                    run_command(f"rpm2cpio {local_path.name} | cpio -idmv", cwd=temp_dir)

            miopen_src_paths = list(glob.glob(str(temp_dir / "opt/rocm-*/share/miopen")))
            if not miopen_src_paths:
                print("    No kernel files were extracted. Nothing to copy.")
                return

            target_dir = torch_path / "torch/share/miopen"
            print(f"    Copying kernel files to {target_dir}...")
            shutil.copytree(miopen_src_paths[0], target_dir, dirs_exist_ok=True)
            print("[OK] MIOpen kernels installed successfully.")

        finally:
            shutil.rmtree(temp_dir)

def _detect_environment_config(pip_executable: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Detect PyTorch version, ROCm version, and Python version from the environment."""
    try:
        # Get PyTorch version
        result = subprocess.check_output(
            [pip_executable, 'show', 'torch'],
            text=True,
            stderr=subprocess.DEVNULL
        )
        pytorch_version = None
        for line in result.splitlines():
            if line.startswith('Version:'):
                pytorch_version = line.split(':', 1)[1].strip()
                # Convert to major.minor format (e.g., 2.5.0 -> 2.5)
                pytorch_version = '.'.join(pytorch_version.split('.')[:2])
                break

        # Get ROCm version from PyTorch
        try:
            hip_version = subprocess.check_output(
                f"{pip_executable.replace('pip', 'python')} -c 'import torch; print(torch.version.hip if hasattr(torch.version, \"hip\") else \"\")'",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()

            rocm_version = None
            if hip_version and hip_version != "None":
                # HIP version format: 6.2.41134 -> extract 6.2
                rocm_version = '.'.join(hip_version.split('.')[:2])
        except:
            rocm_version = None

        # Get Python version
        python_version = subprocess.check_output(
            [pip_executable.replace('pip', 'python'), '--version'],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        # Extract version (e.g., "Python 3.11.5" -> "3.11")
        python_version = '.'.join(python_version.split()[1].split('.')[:2])

        return pytorch_version, rocm_version, python_version
    except:
        return None, None, None

def _find_flash_attn_wheel(arch: str, pip_executable: str) -> Optional[str]:
    """Find the appropriate Flash Attention wheel from FlashAttentionDist repository."""

    # Detect environment configuration
    pytorch_version, rocm_version, python_version = _detect_environment_config(pip_executable)

    if not all([pytorch_version, rocm_version, python_version]):
        print("[WARN] Could not detect environment configuration")
        print(f"   PyTorch: {pytorch_version}, ROCm: {rocm_version}, Python: {python_version}")
        return None

    print(f"[INFO] Detected configuration:")
    print(f"   Architecture: {arch}")
    print(f"   ROCm: {rocm_version}")
    print(f"   PyTorch: {pytorch_version}")
    print(f"   Python: {python_version}")

    # Construct wheel URL based on detected configuration
    # Format: wheels/{arch}/rocm{ver}/pytorch{ver}/python{ver}/
    base_url = "https://github.com/TesslateAI/FlashAttentionDist/raw/main"
    wheel_dir = f"wheels/{arch}/rocm{rocm_version}/pytorch{pytorch_version}/python{python_version}"

    # First, try to get the index.json to find available wheels
    index_url = f"{base_url}/wheels/index.json"
    try:
        response = requests.get(index_url, timeout=10)
        if response.status_code == 200:
            index = response.json()

            # Navigate through the index structure
            if (arch in index.get('architectures', {}) and
                f"rocm{rocm_version}" in index['architectures'][arch] and
                f"pytorch{pytorch_version}" in index['architectures'][arch][f"rocm{rocm_version}"]):

                wheels = index['architectures'][arch][f"rocm{rocm_version}"][f"pytorch{pytorch_version}"]

                # Find wheel for the correct Python version
                for wheel_info in wheels:
                    if wheel_info.get('python') == f"python{python_version}":
                        wheel_path = wheel_info['path']
                        wheel_url = f"{base_url}/wheels/{wheel_path}"

                        # Verify the wheel exists
                        verify_response = requests.head(wheel_url, timeout=5)
                        if verify_response.status_code == 200:
                            print(f"[OK] Found pre-built wheel: {wheel_info['filename']}")
                            return wheel_url
    except Exception as e:
        print(f"[WARN] Could not fetch wheel index: {e}")

    # Fallback: try to construct URL directly and check if it exists
    # Try to find any wheel in the directory by checking common patterns
    print(f"[INFO] Searching for wheel at: {wheel_dir}")
    return None

def patch_rocm_environment(arch="gfx942", venv_path: str = None, rocm_version="latest", install_kernels=True, build_from_source=False, pytorch_install=None):
    """Automates patching for the ROCm environment by reading from a dependency file."""
    
    # Correctly determine the python and pip executables
    if venv_path:
        venv_path_obj = Path(venv_path).resolve()
        python_executable = str(venv_path_obj / "bin/python")
        pip_executable = str(venv_path_obj / "bin/pip")
        if not Path(python_executable).exists():
            print(f"[ERROR] Python executable not found at '{python_executable}'.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Targeting explicit virtual environment at: {venv_path_obj}")
    elif sys.prefix != sys.base_prefix:
        # A virtual environment is active, trust sys.executable
        python_executable = sys.executable
        pip_executable = str(Path(sys.executable).parent / "pip")
        print(f"[INFO] Targeting active virtual environment at: {sys.prefix}")
    else:
        # Fallback to the global/system environment
        python_executable = sys.executable
        pip_executable = str(Path(sys.executable).parent / "pip")
        print(f"[INFO] Targeting current/global Python environment.")
        print("[WARN] It is highly recommended to use a virtual environment.", file=sys.stderr)

    if shutil.which('git') is None:
        print("[ERROR] 'git' command not found. Please install Git.", file=sys.stderr)
        sys.exit(1)
    
    deps_path = Path(__file__).parent.parent / 'patcher-deps.json'
    with open(deps_path, 'r') as f:
        deps = json.load(f)

    build_deps = " ".join(f'"{p}"' for p in deps['build_from_source']['packages'])
    ml_deps = " ".join(f'"{p}"' for p in deps['machine_learning']['packages'])
    
    print("\n--- 1. Installing Dependencies into Target Environment ---")
    run_command(f"{pip_executable} install --upgrade pip")
    
    # Handle PyTorch installation based on pytorch_install parameter
    if pytorch_install:
        print("\n--- Installing PyTorch for ROCm ---")
        pytorch_deps = deps['pytorch_rocm']
        
        if pytorch_install == "stable":
            print(f"   -> Installing PyTorch STABLE for ROCm")
            index_url = pytorch_deps['stable_index_url']
            pytorch_packages = " ".join(pytorch_deps['packages'])
            run_command(f"{pip_executable} install {pytorch_packages} --index-url {index_url}")
        elif pytorch_install == "nightly":
            print(f"   -> Installing PyTorch NIGHTLY for ROCm 6.4")
            index_url = pytorch_deps['nightly_index_url']
            pytorch_packages = " ".join(pytorch_deps['packages'])
            run_command(f"{pip_executable} install --pre {pytorch_packages} --index-url {index_url}")
        elif pytorch_install.startswith("http"):
            # Direct wheel URL provided
            print(f"   -> Installing PyTorch from provided wheel URL")
            run_command(f"{pip_executable} install {pytorch_install}")
        else:
            print(f"[ERROR] Invalid --install-pytorch value: {pytorch_install}", file=sys.stderr)
            print("Use 'stable', 'nightly', or a direct wheel URL.", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n--- Skipping PyTorch installation ---")
        print("   -> Use --install-pytorch [stable|nightly|<wheel-url>] to install PyTorch")
    
    if build_from_source:
        print("   -> Installing build-time dependencies...")
        run_command(f"{pip_executable} install {build_deps}")
    
    print("   -> Installing core machine learning libraries...")
    run_command(f"{pip_executable} install {ml_deps}")

    if install_kernels:
        torch_install_path = _get_pytorch_path(pip_executable)
        gfx_archs = [a.strip() for a in arch.split(';')]
        _install_miopen_kernels(torch_install_path, rocm_version, gfx_archs)
    else:
        print("\n--- Skipping MIOpen Kernel installation as requested. ---")

    # Flash Attention installation
    print(f"\n--- Installing Flash Attention 2 ---")

    if build_from_source:
        print(f"[INFO] Building Flash Attention from source for {arch}")
        if os.path.exists("flash-attention"): shutil.rmtree("flash-attention")
        run_command("git clone https://github.com/ROCm/flash-attention.git")
        build_command = f"MAX_JOBS=$(nproc) GPU_ARCHS='{arch}' {python_executable} setup.py install"
        run_command(build_command, cwd="flash-attention")
        shutil.rmtree("flash-attention", ignore_errors=True)
    else:
        print(f"[INFO] Looking for pre-built Flash Attention wheel...")

        # Try to find a matching wheel from FlashAttentionDist
        wheel_url = _find_flash_attn_wheel(arch, pip_executable)

        if wheel_url:
            print(f"[INFO] Installing pre-built Flash Attention wheel")
            try:
                run_command(f"{pip_executable} install {wheel_url}")
                print("[OK] Pre-built wheel installed successfully")
            except SystemExit:
                print(f"\n{'='*70}")
                print(f"[ERROR] Failed to install pre-built wheel")
                print(f"{'='*70}")
                wheel_url = None

        if not wheel_url:
            # No wheel found or installation failed - build from source
            print(f"\n{'='*70}")
            print(f"[WARN] No pre-built Flash Attention wheel available for your configuration")
            print(f"[WARN] Architecture: {arch}")

            pytorch_ver, rocm_ver, python_ver = _detect_environment_config(pip_executable)
            if pytorch_ver:
                print(f"[WARN] PyTorch: {pytorch_ver} | ROCm: {rocm_ver} | Python: {python_ver}")

            print(f"\n[INFO] Building Flash Attention from source instead...")
            print(f"[INFO] This may take 10-30 minutes depending on your system")
            print(f"{'='*70}\n")

            if os.path.exists("flash-attention"):
                shutil.rmtree("flash-attention")

            run_command("git clone https://github.com/ROCm/flash-attention.git")
            build_command = f"MAX_JOBS=$(nproc) GPU_ARCHS='{arch}' {python_executable} setup.py install"
            run_command(build_command, cwd="flash-attention")
            shutil.rmtree("flash-attention", ignore_errors=True)
    
# --- Verifying Installation in Target Environment ---
    print("\n--- Verifying Installation in Target Environment ---")
    # FIX: Use double quotes inside the python f-string to avoid conflicting with the outer single quotes for the shell command.
    verify_script = 'import importlib; importlib.invalidate_caches(); import flash_attn; print(f"OK - Successfully imported flash_attn version: {flash_attn.__version__}")'
    try:
        run_command(f"{python_executable} -c '{verify_script}'")
        print("\n[SUCCESS] Patching complete. Your environment is ready.")
    except SystemExit:
        print(f"\n[ERROR] Verification failed. Could not import a required library.", file=sys.stderr)
        sys.exit(1)