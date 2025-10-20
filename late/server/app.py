from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from waitress import serve
from late.engine.queue_manager import QueueManager
import os
import yaml
import subprocess
import threading
import time
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages
qm = QueueManager()

# Add custom Jinja2 filters
@app.template_filter('basename')
def basename_filter(path):
    """Extract filename from path"""
    return os.path.basename(path)

# Directory to persist run data
RUNS_DIR = Path.cwd() / "training_runs"
RUNS_DIR.mkdir(exist_ok=True)
RUNS_INDEX_FILE = RUNS_DIR / ".runs_index.json"

# Store running jobs (in-memory, will reset on server restart)
running_jobs = {}

# Default configuration values
DEFAULT_CONFIG = {
    'output_dir': './outputs/',
    'training_type': 'lora',
    'max_seq_length': 2048,
    'loss_masking_strategy': 'full',
    'batch_size': 4,
    'gradient_accumulation': 4,
    'epochs': 3,
    'learning_rate': 2e-4,
    'lr_scheduler_type': 'cosine',
    'optim': 'adamw_torch_fused',
    'save_steps': 100,
    'gradient_checkpointing': True,
    'torch_compile': True,
    'tf32': True,
    'cache_dir': '~/.cache/late/models',
    'report_to_wandb': False,
    'upload_to_hub': False,
    'use_unsloth': False,
    'lora': {
        'r': 64,
        'lora_alpha': 128,
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    }
}

def load_runs_index():
    """Load the runs index from disk"""
    if RUNS_INDEX_FILE.exists():
        with open(RUNS_INDEX_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_runs_index(runs_index):
    """Save the runs index to disk"""
    with open(RUNS_INDEX_FILE, 'w') as f:
        json.dump(runs_index, f, indent=2)

def save_run_data(run_id, run_data):
    """Save run data to a JSON file"""
    run_file = RUNS_DIR / f"{run_id}.json"
    # Convert log list to avoid huge files - keep last 1000 lines
    data_to_save = run_data.copy()
    if 'log' in data_to_save and len(data_to_save['log']) > 1000:
        data_to_save['log'] = data_to_save['log'][-1000:]
    # Remove process object
    if 'process' in data_to_save:
        del data_to_save['process']

    with open(run_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    # Update index
    runs_index = load_runs_index()
    runs_index[run_id] = {
        'started_at': run_data.get('started_at'),
        'status': run_data.get('status'),
        'config_path': run_data.get('config_path'),
        'last_updated': time.time()
    }
    save_runs_index(runs_index)

def load_run_data(run_id):
    """Load run data from disk"""
    run_file = RUNS_DIR / f"{run_id}.json"
    if run_file.exists():
        with open(run_file, 'r') as f:
            return json.load(f)
    return None

def get_config_with_defaults(config_path):
    """Load config and mark which values are defaults"""
    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f) or {}

    config_with_defaults = {}

    for key, default_value in DEFAULT_CONFIG.items():
        if key in user_config:
            if isinstance(default_value, dict):
                # Handle nested configs like lora
                config_with_defaults[key] = {
                    'value': user_config[key],
                    'is_default': user_config[key] == default_value,
                    'default': default_value
                }
            else:
                config_with_defaults[key] = {
                    'value': user_config[key],
                    'is_default': user_config[key] == default_value,
                    'default': default_value
                }
        else:
            config_with_defaults[key] = {
                'value': default_value,
                'is_default': True,
                'default': default_value
            }

    # Add any additional keys from user config that aren't in defaults
    for key, value in user_config.items():
        if key not in config_with_defaults:
            config_with_defaults[key] = {
                'value': value,
                'is_default': False,
                'default': None
            }

    return config_with_defaults

# Load existing runs on startup
def init_runs():
    """Initialize runs from disk"""
    global running_jobs
    runs_index = load_runs_index()
    for run_id, run_info in runs_index.items():
        run_data = load_run_data(run_id)
        if run_data:
            running_jobs[run_id] = run_data

init_runs()

# Lazy import for training script generation (requires torch)
def get_training_script_generator():
    """Lazy import of generate_training_script to avoid requiring torch for dashboard."""
    try:
        from late.engine.training import generate_training_script
        return generate_training_script
    except ImportError as e:
        flash(f"Training dependencies not installed. Install with: pip install -e '.[training]'", "error")
        raise

@app.route('/')
def index():
    queues = qm.list_queues()
    queue_data = {}
    for q_name in queues:
        queue_data[q_name] = qm.get_queue_contents(q_name)
    return render_template('index.html', queue_data=queue_data, running_jobs=running_jobs)

@app.route('/configs')
def configs():
    """Show all available config files"""
    config_dir = Path.cwd() / "examples"
    configs = []
    if config_dir.exists():
        for config_file in config_dir.rglob("*.yml"):
            configs.append({
                'path': str(config_file),
                'name': config_file.name,
                'relative': str(config_file.relative_to(Path.cwd()))
            })

    # Count configs by type for stats
    example_count = sum(1 for c in configs if c['relative'].startswith('examples'))
    custom_count = sum(1 for c in configs if c['relative'].startswith('configs'))

    return render_template('configs.html',
                         configs=configs,
                         example_count=example_count,
                         custom_count=custom_count)

@app.route('/config/create', methods=['GET', 'POST'])
def create_config():
    """Create new config file from form"""
    if request.method == 'GET':
        return render_template('create_config.html')

    # Process form submission
    config = {}

    # Basic settings
    config['base_model'] = request.form.get('base_model', '')
    config['dataset_name'] = request.form.get('dataset_name', '')
    config['output_model_name'] = request.form.get('output_model_name', '')
    config['output_dir'] = request.form.get('output_dir', './outputs/')

    # Training type
    config['training_type'] = request.form.get('training_type', 'lora')
    config['max_seq_length'] = int(request.form.get('max_seq_length', 2048))

    # Unsloth acceleration
    config['use_unsloth'] = request.form.get('use_unsloth') == 'on'

    # Loss masking
    config['loss_masking_strategy'] = request.form.get('loss_masking_strategy', 'full')

    # Hyperparameters
    config['batch_size'] = int(request.form.get('batch_size', 4))
    config['gradient_accumulation'] = int(request.form.get('gradient_accumulation', 4))
    config['epochs'] = int(request.form.get('epochs', 3))
    config['learning_rate'] = float(request.form.get('learning_rate', 2e-4))
    config['lr_scheduler_type'] = request.form.get('lr_scheduler_type', 'cosine')
    config['optim'] = request.form.get('optim', 'adamw_torch_fused')
    config['save_steps'] = int(request.form.get('save_steps', 100))

    # Memory optimization
    config['gradient_checkpointing'] = request.form.get('gradient_checkpointing') == 'on'
    config['torch_compile'] = request.form.get('torch_compile') == 'on'
    config['tf32'] = request.form.get('tf32') == 'on'
    config['cache_dir'] = request.form.get('cache_dir', '~/.cache/late/models')

    # LoRA settings (if LoRA training)
    if config['training_type'] == 'lora':
        config['lora'] = {
            'r': int(request.form.get('lora_r', 64)),
            'lora_alpha': int(request.form.get('lora_alpha', 128)),
            'lora_dropout': float(request.form.get('lora_dropout', 0.05)),
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
        }

    # Logging
    config['report_to_wandb'] = request.form.get('report_to_wandb') == 'on'
    config['upload_to_hub'] = request.form.get('upload_to_hub') == 'on'

    # Optional settings
    ntfy_topic = request.form.get('ntfy_topic', '').strip()
    if ntfy_topic:
        config['ntfy_topic'] = ntfy_topic

    hf_token = request.form.get('hf_token', '').strip()
    if hf_token:
        config['hf_token'] = hf_token

    # Save config file
    config_name = request.form.get('config_name', 'new_config.yml')
    if not config_name.endswith('.yml'):
        config_name += '.yml'

    config_path = Path.cwd() / 'configs' / config_name
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    flash(f"Config '{config_name}' created successfully at {config_path}!", "success")
    return redirect(url_for('configs'))

@app.route('/queue/create', methods=['POST'])
def create_queue():
    queue_name = request.form.get('queue_name')
    if queue_name:
        qm.create_queue(queue_name)
        flash(f"Queue '{queue_name}' created successfully!", "success")
    else:
        flash("Queue name cannot be empty.", "error")
    return redirect(url_for('index'))

@app.route('/queue/delete', methods=['POST'])
def delete_queue():
    queue_name = request.form.get('queue_name')
    qm.delete_queue(queue_name)
    flash(f"Queue '{queue_name}' deleted.", "info")
    return redirect(url_for('index'))

@app.route('/queue/add_job', methods=['POST'])
def add_job_to_queue():
    queue_name = request.form.get('queue_name')
    config_path = request.form.get('config_path')
    if not os.path.exists(config_path):
        flash(f"Error: Config file not found at '{config_path}'.", "error")
    else:
        qm.add_to_queue(queue_name, config_path)
        flash(f"Added '{config_path}' to '{queue_name}'.", "success")
    return redirect(url_for('index'))

@app.route('/train/start', methods=['POST'])
def start_training():
    """Start a training job directly from the dashboard"""
    config_path = request.form.get('config_path')

    if not os.path.exists(config_path):
        flash(f"Error: Config file not found at '{config_path}'.", "error")
        return redirect(url_for('index'))

    # Generate unique job ID
    job_id = f"job_{int(time.time())}"

    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Generate training script (lazy import)
    try:
        generate_training_script = get_training_script_generator()
        script_content = generate_training_script(config)
    except ImportError:
        return redirect(url_for('index'))

    # Save training script to temp file
    script_path = Path(config.get('output_dir', './outputs')) / f"{job_id}_train.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # Start training in background thread
    def run_training():
        try:
            process = subprocess.Popen(
                ['python', str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            running_jobs[job_id]['process'] = process
            running_jobs[job_id]['status'] = 'running'
            save_run_data(job_id, running_jobs[job_id])

            # Stream output and save periodically
            line_count = 0
            for line in process.stdout:
                running_jobs[job_id]['log'].append(line.rstrip())
                line_count += 1

                # Save every 50 lines
                if line_count % 50 == 0:
                    save_run_data(job_id, running_jobs[job_id])

            process.wait()
            running_jobs[job_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
            running_jobs[job_id]['return_code'] = process.returncode
            save_run_data(job_id, running_jobs[job_id])

        except Exception as e:
            running_jobs[job_id]['status'] = 'error'
            running_jobs[job_id]['error'] = str(e)
            save_run_data(job_id, running_jobs[job_id])

    # Initialize job tracking
    running_jobs[job_id] = {
        'config_path': config_path,
        'script_path': str(script_path),
        'status': 'starting',
        'log': [],
        'started_at': time.time()
    }

    # Save initial run data
    save_run_data(job_id, running_jobs[job_id])

    # Start background thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    flash(f"Training job '{job_id}' started! View at /run/{job_id}", "success")
    return redirect(url_for('run_details', run_id=job_id))

@app.route('/jobs')
def jobs():
    """Show list of all training runs"""
    runs_index = load_runs_index()

    # Convert to list and sort by start time (newest first)
    runs_list = []
    for run_id, run_info in runs_index.items():
        # Get live status from running_jobs if available
        if run_id in running_jobs:
            status = running_jobs[run_id]['status']
        else:
            status = run_info['status']

        runs_list.append({
            'id': run_id,
            'status': status,
            'config_path': run_info['config_path'],
            'started_at': run_info['started_at'],
            'last_updated': run_info.get('last_updated', run_info['started_at'])
        })

    runs_list.sort(key=lambda x: x['started_at'], reverse=True)

    return render_template('jobs_list.html', runs=runs_list, now=time.time())

@app.route('/run/<run_id>')
def run_details(run_id):
    """Show detailed view of a single training run"""
    # Try to get from memory first, then disk
    if run_id in running_jobs:
        job = running_jobs[run_id]
    else:
        job = load_run_data(run_id)
        if not job:
            flash(f"Run '{run_id}' not found", "error")
            return redirect(url_for('jobs'))

    # Load config with defaults
    config_with_defaults = None
    if os.path.exists(job['config_path']):
        try:
            config_with_defaults = get_config_with_defaults(job['config_path'])
        except Exception as e:
            flash(f"Error loading config: {e}", "error")

    return render_template('run_details.html',
                         job_id=run_id,
                         job=job,
                         config_with_defaults=config_with_defaults,
                         now=time.time())

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    """API endpoint to get job status (for live updates)"""
    if job_id not in running_jobs:
        # Try loading from disk
        job = load_run_data(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
    else:
        job = running_jobs[job_id]

    return jsonify({
        'id': job_id,
        'status': job['status'],
        'config_path': job['config_path'],
        'log': job['log'][-50:],  # Last 50 lines
        'started_at': job['started_at']
    })

@app.route('/api/job/<job_id>/stop', methods=['POST'])
def stop_job(job_id):
    """Stop a running job"""
    if job_id not in running_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = running_jobs[job_id]
    if 'process' in job and job['process'].poll() is None:
        job['process'].terminate()
        job['status'] = 'stopped'
        return jsonify({'success': True, 'message': 'Job stopped'})

    return jsonify({'error': 'Job is not running'}), 400

def run_server(host='0.0.0.0', port=8080):
    # Use Waitress, a production-ready WSGI server
    print(f"Starting Late Training Dashboard at http://{host}:{port}")
    print(f"Open your browser and navigate to http://localhost:{port}")
    serve(app, host=host, port=port)