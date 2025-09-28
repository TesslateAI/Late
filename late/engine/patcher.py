import subprocess
import sys
import shutil
import os
import re
import glob
import requests
import json
from pathlib import Path

def run_command(command: str, cwd: str = None):
    """Executes a shell command and streams its output."""
    print(f"🔩 Executing: {command}")
    process = subprocess.Popen(
        command, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"\n{'='*60}\n❌ ERROR: Command failed with exit code {process.returncode}\n❌ Command: {command}\n{'='*60}", file=sys.stderr)
        sys.exit(process.returncode)

def _get_pytorch_path(pip_executable: str) -> Path:
    """Finds the installation path of PyTorch in the target environment."""
    try:
        print("🔍 Finding PyTorch installation path...")
        result = subprocess.check_output([pip_executable, 'show', 'torch'], text=True)
        for line in result.splitlines():
            if line.startswith('Location:'):
                path = Path(line.split(':', 1)[1].strip())
                print(f"✅ Found PyTorch at: {path}")
                return path
        raise FileNotFoundError("Could not parse 'Location:' from pip output.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FATAL ERROR: PyTorch is not installed or could not be found in the target environment.", file=sys.stderr)
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
        print("⚠️ WARNING: Unsupported OS for MIOpen kernel download. Skipping.", file=sys.stderr)
        return

    print(f"OS Detected: {distro.capitalize()} {version_id}")
    
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
                    print(f"    ⚠️ WARNING: No MIOpen kernel package found for {arch} at {repo_url}. Skipping.", file=sys.stderr)
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
            print("✅ MIOpen kernels installed successfully.")

        finally:
            shutil.rmtree(temp_dir)

def patch_rocm_environment(arch="gfx942", venv_path: str = None, rocm_version="latest", install_kernels=True, build_from_source=False):
    """Automates patching for the ROCm environment by reading from a dependency file."""
    
    # --- FIX 1: Correctly determine the python and pip executables ---
    if venv_path:
        # User has provided an explicit path to a venv
        venv_path_obj = Path(venv_path).resolve()
        python_executable = str(venv_path_obj / "bin/python")
        pip_executable = str(venv_path_obj / "bin/pip")
        if not Path(python_executable).exists():
            print(f"❌ FATAL ERROR: Python executable not found at '{python_executable}'.", file=sys.stderr)
            sys.exit(1)
        print(f"🎯 Targeting explicit virtual environment at: {venv_path_obj}")
    elif 'VIRTUAL_ENV' in os.environ:
        # We are running inside an activated venv
        venv_path_obj = Path(os.environ['VIRTUAL_ENV']).resolve()
        python_executable = str(venv_path_obj / "bin/python")
        pip_executable = str(venv_path_obj / "bin/pip")
        print(f"🎯 Targeting active virtual environment at: {venv_path_obj}")
    else:
        # Fallback to the global/system environment
        python_executable = sys.executable
        pip_executable = f"{Path(sys.executable).parent}/pip"
        print(f"🎯 Targeting current/global Python environment.")
        print("   -> WARNING: It is highly recommended to use a virtual environment.", file=sys.stderr)

    if shutil.which('git') is None:
        print("❌ FATAL ERROR: 'git' command not found. Please install Git.", file=sys.stderr)
        sys.exit(1)
    
    deps_path = Path(__file__).parent.parent / 'patcher-deps.json'
    with open(deps_path, 'r') as f:
        deps = json.load(f)

    build_deps = " ".join(f'"{p}"' for p in deps['build_from_source']['packages'])
    ml_deps = " ".join(f'"{p}"' for p in deps['machine_learning']['packages'])
    
    print("\n--- 1. Installing Dependencies into Target Environment ---")
    run_command(f"{pip_executable} install --upgrade pip")
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

    if build_from_source:
        print(f"\n--- Building Flash Attention from source for {arch} ---")
        if os.path.exists("flash-attention"): shutil.rmtree("flash-attention")
        run_command("git clone https://github.com/ROCm/flash-attention.git")
        build_command = f"MAX_JOBS=$(nproc) GPU_ARCHS='{arch}' {python_executable} setup.py install"
        run_command(build_command, cwd="flash-attention")
        shutil.rmtree("flash-attention", ignore_errors=True)
    else:
        print(f"\n--- Installing Flash Attention from pre-built wheel ---")
        wheel_url = "https://github.com/TesslateAI/FlashAttentionDist/raw/main/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
        run_command(f"{pip_executable} install {wheel_url}")
    
    # --- FIX 2: Make the verification script ASCII-safe to avoid SyntaxError ---
    print("\n--- Verifying Installation in Target Environment ---")
    verify_script = (
        "import importlib; "
        "importlib.invalidate_caches(); "
        "import flash_attn; "
        "print(f'OK - Successfully imported flash_attn version: {flash_attn.__version__}')"
    )
    try:
        run_command(f"{python_executable} -c '{verify_script}'")
        print("\n🎉 Patching complete. Your environment is ready.")
    except SystemExit:
        print(f"\n❌ VERIFICATION FAILED. Could not import a required library.", file=sys.stderr)
        sys.exit(1)