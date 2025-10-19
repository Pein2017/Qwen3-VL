"""Pure YAML config loader - directly instantiates ms-swift objects"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Set, List

from swift.llm.argument import TrainArguments

from .prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, USER_PROMPT


class ConfigLoader:
    """Load YAML config and directly instantiate ms-swift dataclasses.
    
    No CLI argument parsing - direct object construction from YAML.
    All hyperparameters must be explicitly defined in YAML.
    """
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML file into dictionary.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def _normalize_to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    @staticmethod
    def load_yaml_with_extends(config_path: str, _visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Load YAML and resolve inheritance via 'extends'/'inherit'.
        
        Supports a top-level key in the YAML:
          - extends: str | list[str]     # relative to the current file
          - inherit: str | list[str]     # alias of extends
        
        Bases are merged in order (earlier are lower precedence).
        The current file has the highest precedence.
        Cycles are detected and will raise a ValueError.
        """
        abs_path = str(Path(config_path).resolve())
        visited: Set[str] = set(_visited or set())
        if abs_path in visited:
            raise ValueError(f"Cyclic config inheritance detected at: {abs_path}")
        visited.add(abs_path)

        current_dir = Path(abs_path).parent
        config = ConfigLoader.load_yaml(abs_path) or {}

        # Gather base paths from supported keys
        extends_value = None
        if isinstance(config, dict):
            extends_value = config.pop('extends', None)
            if extends_value is None:
                extends_value = config.pop('inherit', None)

        base_paths = ConfigLoader._normalize_to_list(extends_value)

        # Merge all bases in order
        merged_base: Dict[str, Any] = {}
        for base_ref in base_paths:
            base_path = Path(base_ref)
            if not base_path.is_absolute():
                base_path = (current_dir / base_path).resolve()
            base_cfg = ConfigLoader.load_yaml_with_extends(str(base_path), visited)
            merged_base = ConfigLoader.merge_configs(merged_base, base_cfg)

        # Finally merge current file on top
        return ConfigLoader.merge_configs(merged_base, config)
    
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Deep merge two config dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    @staticmethod
    def resolve_prompts(config: Dict) -> Dict:
        """Resolve prompts to simple strings from config or fallbacks.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved prompts
        """
        prompts_config = config.get('prompts', {})

        # Prefer external files via YAML if provided; otherwise fall back to module defaults
        scheme = prompts_config.get('scheme', 'A').upper()
        default_system = SYSTEM_PROMPT_A if scheme == 'A' else SYSTEM_PROMPT_B
        system_prompt = prompts_config.get('system', default_system)
        user_prompt = prompts_config.get('user', USER_PROMPT)

        # If user prompt is provided as a path to a text file, load contents
        try:
            if isinstance(user_prompt, str) and user_prompt.endswith('.txt') and os.path.isfile(user_prompt):
                with open(user_prompt, 'r', encoding='utf-8') as f:
                    user_prompt = f.read().strip('\n')
        except Exception:
            pass
        
        # Store resolved prompts in appropriate sections
        if 'custom' not in config:
            config['custom'] = {}
        config['custom']['user_prompt'] = user_prompt
        
        if 'template' not in config:
            config['template'] = {}
        # Map to TemplateArguments.system
        config['template']['system'] = system_prompt
        
        return config
    
    @staticmethod
    def build_train_arguments(config: Dict) -> TrainArguments:
        """Directly instantiate TrainArguments from config.
        
        TrainArguments is a unified dataclass that inherits from:
        - Seq2SeqTrainingArguments (HuggingFace Transformers)
        - TunerArguments (LoRA, adapters, etc.)
        - DataArguments (dataset configuration)
        - ModelArguments (model loading)
        - QuantizeArguments (quantization)
        - TemplateArguments (prompt templates)
        - SwanlabArguments (logging)
        
        We merge all config sections and pass to TrainArguments constructor,
        which will use ms-swift's built-in defaults for any missing fields.
        
        Args:
            config: Configuration dictionary from YAML
            
        Returns:
            Fully initialized TrainArguments object
        """
        # Collect all arguments from different sections
        args_dict = {}
        
        # Merge sections in order (later can override earlier)
        sections = ['model', 'quantization', 'data', 'template', 'tuner', 'training']
        
        for section in sections:
            if section in config:
                section_data = config[section]
                if isinstance(section_data, dict):
                    # Flatten nested config into single dict for TrainArguments
                    args_dict.update(section_data)
        
        # Handle DeepSpeed configuration
        if 'deepspeed' in config:
            ds_config = config['deepspeed']
            if ds_config.get('enabled', False):
                args_dict['deepspeed'] = ds_config.get('config', 'zero2')
        
        # Create TrainArguments with all merged parameters
        # ms-swift's TrainArguments will fill in defaults for missing fields
        train_args = TrainArguments(**args_dict)
        
        return train_args
    
    @staticmethod
    def get_custom_config(config: Dict) -> Dict[str, Any]:
        """Extract custom dataset configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Custom configuration dict for dataset creation
        """
        return config.get('custom', {})
    
    @staticmethod
    def load_training_config(
        config_path: str,
        base_config_path: Optional[str] = None
    ) -> tuple[TrainArguments, Dict[str, Any]]:
        """Load configuration and return ms-swift TrainArguments + custom config.
        
        Main entry point for config loading.
        
        Args:
            config_path: Path to main YAML configuration file
            base_config_path: Optional path to base config (for inheritance)
            
        Returns:
            Tuple of (train_args, custom_config)
            - train_args: ms-swift TrainArguments object
            - custom_config: Dict with custom dataset parameters
        """
        # Load main config (with recursive inheritance)
        config = ConfigLoader.load_yaml_with_extends(config_path)
        
        # Optionally merge with base config
        if base_config_path:
            base_config = ConfigLoader.load_yaml_with_extends(base_config_path)
            # CLI base_config is the lowest precedence
            config = ConfigLoader.merge_configs(base_config, config)
        
        # Resolve prompt key references to actual prompts
        config = ConfigLoader.resolve_prompts(config)

        # Apply global max length if provided to avoid ambiguity
        try:
            global_max_len = config.get('global_max_length')
            if isinstance(global_max_len, int) and global_max_len > 0:
                # Ensure sections exist
                config.setdefault('model', {})
                config.setdefault('template', {})
                # Override both model.max_model_len and template.max_length
                config['model']['max_model_len'] = global_max_len
                config['template']['max_length'] = global_max_len
        except Exception:
            pass
        
        # Build TrainArguments object directly (no CLI parsing)
        train_args = ConfigLoader.build_train_arguments(config)
        
        # Extract custom dataset configuration
        custom_config = ConfigLoader.get_custom_config(config)
        
        return train_args, custom_config


__all__ = ['ConfigLoader']

