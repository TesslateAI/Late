import itertools
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy

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
        'timestamp': datetime.now().isoformat()
    }
    
    with open(sweep_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return config_paths, sweep_dir