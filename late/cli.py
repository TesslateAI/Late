import click
import os
from late.engine.patcher import patch_rocm_environment
from late.engine.queue_manager import QueueManager
from late.engine.config import save_token
from late.engine.training import run_training_job
from late.server.app import run_server

# Initialize QueueManager to use a default 'queues' directory
qm = QueueManager()

@click.group()
def cli():
    """A tool for patching, scheduling, and running training jobs on ROCm."""
    pass

@cli.command()
@click.option('--arch', default='gfx942', help='Target GPU architecture (e.g., gfx942).')
def patch(arch):
    """Patches the ROCm environment with Triton and Flash Attention."""
    click.echo(f"🚀 Patching environment for ROCm architecture: {arch}")
    patch_rocm_environment(arch)

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Run a single training job from a YAML config file."""
    run_training_job(config_path)

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
def start_queue(queue_name):
    """Start running all training jobs in a queue sequentially."""
    jobs = qm.get_queue_contents(queue_name)
    if not jobs:
        click.echo(f"Queue '{queue_name}' is empty or does not exist.")
        return
    
    click.echo(f"▶️ Starting queue '{queue_name}' with {len(jobs)} jobs.")
    for i, config_path in enumerate(jobs):
        click.echo(f"\n--- Running job {i+1}/{len(jobs)} ---")
        if not os.path.exists(config_path):
            click.echo(f"⚠️ Warning: Config file not found, skipping: {config_path}", err=True)
            continue
        run_training_job(config_path)
    
    click.echo(f"\n🎉 Queue '{queue_name}' finished.")

@cli.command()
@click.argument('key', type=click.Choice(['wandb', 'hf_token'], case_sensitive=False))
@click.argument('token')
def set(key, token):
    """Set API tokens (wandb, hf_token)."""
    save_token(key, token)

@cli.command()
@click.option('--port', default=8080, help='Port to run the web dashboard on.')
def serve(port):
    """Launch the web dashboard."""
    click.echo(f"🌐 Launching web dashboard at http://0.0.0.0:{port}")
    click.echo("Use Ctrl+C to shut down.")
    run_server(port=port)

if __name__ == '__main__':
    cli()