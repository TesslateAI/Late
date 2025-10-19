from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from waitress import serve
from late.engine.queue_manager import QueueManager
import os
import yaml
import subprocess
import threading
import time
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages
qm = QueueManager()

# Store running jobs (in-memory, will reset on server restart)
running_jobs = {}

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

            # Stream output
            for line in process.stdout:
                running_jobs[job_id]['log'].append(line.rstrip())

            process.wait()
            running_jobs[job_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
            running_jobs[job_id]['return_code'] = process.returncode

        except Exception as e:
            running_jobs[job_id]['status'] = 'error'
            running_jobs[job_id]['error'] = str(e)

    # Initialize job tracking
    running_jobs[job_id] = {
        'config_path': config_path,
        'script_path': str(script_path),
        'status': 'starting',
        'log': [],
        'started_at': time.time()
    }

    # Start background thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    flash(f"Training job '{job_id}' started! Check the Jobs tab for status.", "success")
    return redirect(url_for('jobs'))

@app.route('/jobs')
def jobs():
    """Show all running and completed jobs"""
    return render_template('jobs.html', jobs=running_jobs)

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    """API endpoint to get job status (for live updates)"""
    if job_id not in running_jobs:
        return jsonify({'error': 'Job not found'}), 404

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