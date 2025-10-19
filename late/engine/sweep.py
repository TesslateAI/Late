import itertools
import yaml
import json
import os
import csv
import math
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SweepConfig:
    """Handles sweep configuration parsing and generation."""

    def __init__(self, base_config_path: str, sweep_config: Dict[str, Any]):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.sweep_config = sweep_config
        self.sweep_id = sweep_config.get('sweep_id', f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def generate_sweep_configs(self) -> List[Dict[str, Any]]:
        """Generate all sweep configurations based on parameter combinations."""
        # Extract sweep parameters and their values
        sweep_params = self.sweep_config.get('sweep_parameters', {})
        overrides = self.sweep_config.get('overrides', {})

        # Get early stopping config
        self.early_stop = self.sweep_config.get('early_stop', {})

        if not sweep_params:
            raise ValueError("No sweep parameters specified")

        # Create all combinations
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]

        configs = []
        for i, values in enumerate(itertools.product(*param_values)):
            # Start with base config
            config = copy.deepcopy(self.base_config)

            # Apply overrides first (like shorter context length for sweeps)
            for key, value in overrides.items():
                self._set_nested_value(config, key, value)

            # Apply sweep parameters
            for param_name, param_value in zip(param_names, values):
                self._set_nested_value(config, param_name, param_value)

            # Add sweep metadata
            config['_sweep_metadata'] = {
                'sweep_id': self.sweep_id,
                'sweep_index': i,
                'sweep_params': dict(zip(param_names, values)),
                'early_stop': self.early_stop
            }

            # Update W&B run name to include sweep info
            if config.get('report_to_wandb'):
                base_name = config['output_model_name'].split('/')[-1]
                sweep_suffix = f"{self.sweep_id}-{i}"
                for param_name, param_value in zip(param_names, values):
                    sweep_suffix += f"-{param_name.replace('_', '')}{param_value}"
                config['_wandb_run_name'] = f"{base_name}-{sweep_suffix}"

            configs.append(config)

        return configs

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested value in config using dot notation (e.g., 'lora.r')."""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

def parse_sweep_file(sweep_file: str) -> Dict[str, Any]:
    """Parse a .sweep file."""
    with open(sweep_file, 'r') as f:
        return yaml.safe_load(f)

def parse_sweep_args(args: List[str]) -> Dict[str, Any]:
    """Parse sweep arguments from command line.
    Format: param1=val1,val2,val3 param2=val4,val5
    """
    sweep_config = {
        'sweep_parameters': {},
        'overrides': {}
    }

    for arg in args:
        if '=' not in arg:
            continue

        key, values_str = arg.split('=', 1)

        # Check if it's an override (starts with override.)
        if key.startswith('override.'):
            override_key = key[9:]  # Remove 'override.' prefix
            # Try to parse as number or boolean
            value = values_str
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            sweep_config['overrides'][override_key] = value
        else:
            # It's a sweep parameter with multiple values
            values = []
            for v in values_str.split(','):
                # Try to parse as number
                try:
                    val = float(v)
                    if val.is_integer():
                        val = int(val)
                    values.append(val)
                except ValueError:
                    values.append(v)

            sweep_config['sweep_parameters'][key] = values

    return sweep_config

def update_comparison_chart(sweep_dir: Path):
    """Finds all run CSVs, plots them on a single chart, and saves it."""
    fig, ax = plt.subplots(figsize=(12, 8))

    run_csv_files = glob.glob(str(sweep_dir / "run_data" / "*.csv"))

    # Sort files to ensure consistent color mapping between runs
    run_csv_files.sort()

    for csv_file in run_csv_files:
        steps, losses = [], []

        # Parse the learning rate (or other param) from the filename
        # Filename format: "run_lr_0.0002.csv" or "run_param_value.csv"
        try:
            filename = os.path.basename(csv_file)
            # Remove 'run_' prefix and '.csv' suffix
            param_info = filename.replace('run_', '').replace('.csv', '')
            # Extract param name and value (e.g., "lr_0.0002" -> "LR = 0.0002")
            if '_' in param_info:
                param_name, param_value = param_info.rsplit('_', 1)
                label = f"{param_name.upper()} = {param_value}"
            else:
                label = param_info
        except (IndexError, ValueError):
            logger.warning(f"Could not parse parameter from filename: {csv_file}. Using filename as label.")
            label = os.path.basename(csv_file).replace('.csv', '')

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                losses.append(float(row['loss']))

        if steps and losses:  # Only plot if there's data
            ax.plot(steps, losses, marker='o', linestyle='-', markersize=3, label=label)

    ax.set_title('Hyperparameter Sweep: Loss Curves', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)

    # Only show legend if there are labeled lines
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    ax.grid(True)

    chart_path = sweep_dir / "loss_comparison_chart.png"
    plt.savefig(chart_path)
    plt.close(fig)
    logger.info(f"📊 Comparison chart updated and saved to {chart_path}")

def create_sweep_configs(base_config_path: str, sweep_source: Any) -> List[Dict[str, Any]]:
    """Create sweep configurations from various sources."""
    if isinstance(sweep_source, str) and sweep_source.endswith('.sweep'):
        # It's a sweep file
        sweep_config = parse_sweep_file(sweep_source)
    elif isinstance(sweep_source, list):
        # It's command line arguments
        sweep_config = parse_sweep_args(sweep_source)
    elif isinstance(sweep_source, dict):
        # It's already a parsed config
        sweep_config = sweep_source
    else:
        raise ValueError(f"Invalid sweep source type: {type(sweep_source)}")

    generator = SweepConfig(base_config_path, sweep_config)
    return generator.generate_sweep_configs()

def save_sweep_configs(configs: List[Dict[str, Any]], output_dir: str = "sweep_runs"):
    """Save generated sweep configs to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    sweep_id = configs[0]['_sweep_metadata']['sweep_id']
    sweep_dir = output_path / sweep_id
    sweep_dir.mkdir(exist_ok=True)

    # Create run_data subdirectory for CSV files
    run_data_dir = sweep_dir / "run_data"
    run_data_dir.mkdir(exist_ok=True)

    # Initialize sweep summary CSV
    summary_csv_path = sweep_dir / "sweep_summary.csv"
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Dynamically create header based on sweep parameters
        param_names = list(configs[0]['_sweep_metadata']['sweep_params'].keys())
        writer.writerow(param_names + ['final_loss', 'best_loss'])

    config_paths = []
    for i, config in enumerate(configs):
        # Remove metadata before saving
        metadata = config.pop('_sweep_metadata')
        wandb_name = config.pop('_wandb_run_name', None)

        # Save config
        config_path = sweep_dir / f"config_{i}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save metadata separately
        metadata_path = sweep_dir / f"config_{i}_metadata.json"
        metadata['config_path'] = str(config_path)
        if wandb_name:
            metadata['wandb_run_name'] = wandb_name
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        config_paths.append(str(config_path))

    # Save sweep summary
    summary = {
        'sweep_id': sweep_id,
        'num_configs': len(configs),
        'config_paths': config_paths,
        'timestamp': datetime.now().isoformat(),
        'sweep_dir': str(sweep_dir)
    }

    with open(sweep_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Sweep configs saved to {sweep_dir}")
    logger.info(f"📊 Sweep summary CSV initialized at {summary_csv_path}")

    return config_paths, sweep_dir
