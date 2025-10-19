"""
Debugging Helpers for Qwen3VL Deep Dive
This module provides utilities for inspecting model internals during execution.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import json


class TensorInspector:
    """Utility class for detailed tensor inspection."""
    
    @staticmethod
    def inspect(name: str, tensor: torch.Tensor, detailed: bool = True) -> Dict[str, Any]:
        """
        Inspect a tensor and print detailed statistics.
        
        Args:
            name: Name/description of the tensor
            tensor: The tensor to inspect
            detailed: Whether to show detailed statistics
            
        Returns:
            Dictionary containing tensor statistics
        """
        if tensor is None:
            print(f"❌ {name}: None")
            return {"status": "None"}
        
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
        }
        
        if detailed and tensor.numel() > 0:
            stats.update({
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "has_nan": torch.isnan(tensor).any().item(),
                "has_inf": torch.isinf(tensor).any().item(),
                "num_elements": tensor.numel(),
                "memory_mb": tensor.element_size() * tensor.numel() / (1024 ** 2),
            })
        
        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"{'='*60}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.6f}")
            else:
                print(f"  {key:20s}: {value}")
        print(f"{'='*60}\n")
        
        return stats
    
    @staticmethod
    def compare_tensors(name1: str, tensor1: torch.Tensor, 
                       name2: str, tensor2: torch.Tensor):
        """Compare two tensors."""
        print(f"\n🔍 Comparing {name1} vs {name2}")
        print(f"  Shape match: {tensor1.shape == tensor2.shape}")
        print(f"  Dtype match: {tensor1.dtype == tensor2.dtype}")
        print(f"  Device match: {tensor1.device == tensor2.device}")
        
        if tensor1.shape == tensor2.shape:
            diff = (tensor1 - tensor2).abs()
            print(f"  Max difference: {diff.max().item():.6e}")
            print(f"  Mean difference: {diff.mean().item():.6e}")
            print(f"  Are equal: {torch.allclose(tensor1, tensor2, rtol=1e-5)}")


class ForwardHookManager:
    """
    Manager for registering forward hooks to inspect intermediate activations.
    """
    
    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []
        
    def register_hook(self, module: nn.Module, name: str):
        """Register a forward hook on a module."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name].append({
                    'shape': tuple(output.shape),
                    'dtype': str(output.dtype),
                    'mean': output.mean().item() if output.numel() > 0 else 0.0,
                    'std': output.std().item() if output.numel() > 0 else 0.0,
                    'output': output.detach().cpu(),
                })
            elif isinstance(output, (tuple, list)):
                # Handle multiple outputs
                self.activations[name].append({
                    'type': 'tuple/list',
                    'length': len(output),
                    'shapes': [tuple(o.shape) if isinstance(o, torch.Tensor) else None 
                              for o in output]
                })
        
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def register_module_hooks(self, model: nn.Module, module_patterns: List[str]):
        """
        Register hooks for modules matching specific patterns.
        
        Args:
            model: The model to hook into
            module_patterns: List of module name patterns (e.g., 'visual.blocks.0')
        """
        for name, module in model.named_modules():
            for pattern in module_patterns:
                if pattern in name:
                    self.register_hook(module, name)
                    print(f"✅ Registered hook: {name}")
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of captured activations."""
        summary = {}
        for name, acts in self.activations.items():
            summary[name] = {
                'num_calls': len(acts),
                'last_shape': acts[-1].get('shape', None) if acts else None,
            }
        return summary
    
    def print_summary(self):
        """Print summary of all captured activations."""
        print(f"\n{'='*70}")
        print("📋 ACTIVATION SUMMARY")
        print(f"{'='*70}")
        
        for name, acts in self.activations.items():
            print(f"\n🔹 {name}")
            print(f"   Calls: {len(acts)}")
            if acts:
                last = acts[-1]
                if 'shape' in last:
                    print(f"   Last shape: {last['shape']}")
                    print(f"   Last mean: {last['mean']:.6f}")
                    print(f"   Last std: {last['std']:.6f}")
                elif 'type' in last:
                    print(f"   Type: {last['type']}")
                    print(f"   Shapes: {last['shapes']}")
        
        print(f"{'='*70}\n")
    
    def save_activations(self, filepath: str):
        """Save activations to file (without tensor data)."""
        summary = {}
        for name, acts in self.activations.items():
            summary[name] = []
            for act in acts:
                # Don't save the actual tensor data
                act_copy = {k: v for k, v in act.items() if k != 'output'}
                summary[name].append(act_copy)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved activation summary to: {filepath}")


class ModelArchitectureInspector:
    """Inspect and print model architecture."""
    
    @staticmethod
    def print_architecture(model: nn.Module, max_depth: int = 3):
        """Print model architecture with indentation."""
        print(f"\n{'='*70}")
        print("🏗️  MODEL ARCHITECTURE")
        print(f"{'='*70}\n")
        
        def print_module(module, name, depth=0, max_depth=max_depth):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            module_name = module.__class__.__name__
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            print(f"{indent}├─ {name} ({module_name})")
            if num_params > 0:
                print(f"{indent}│  └─ Params: {num_params:,} ({trainable:,} trainable)")
            
            # Recursively print children
            for child_name, child_module in module.named_children():
                print_module(child_module, child_name, depth + 1, max_depth)
        
        print_module(model, "model", 0, max_depth)
        print(f"\n{'='*70}\n")
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats = {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
        
        print(f"\n📊 PARAMETER STATISTICS")
        print(f"  Total parameters:     {stats['total']:>15,}")
        print(f"  Trainable parameters: {stats['trainable']:>15,}")
        print(f"  Frozen parameters:    {stats['frozen']:>15,}")
        print()
        
        return stats
    
    @staticmethod
    def find_modules(model: nn.Module, module_type: type) -> List[Tuple[str, nn.Module]]:
        """Find all modules of a specific type."""
        found = []
        for name, module in model.named_modules():
            if isinstance(module, module_type):
                found.append((name, module))
        return found


class ProcessorDebugger:
    """Debug processor operations."""
    
    @staticmethod
    def inspect_processor_output(inputs: Dict[str, torch.Tensor]):
        """Inspect the output from processor."""
        print(f"\n{'='*70}")
        print("🔍 PROCESSOR OUTPUT INSPECTION")
        print(f"{'='*70}\n")
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"📌 {key}")
                print(f"   Shape: {tuple(value.shape)}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Device: {value.device}")
                
                if value.numel() > 0 and value.dtype in [torch.float16, torch.float32, torch.float64]:
                    print(f"   Min: {value.min().item():.4f}")
                    print(f"   Max: {value.max().item():.4f}")
                    print(f"   Mean: {value.mean().item():.4f}")
                
                print()
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def decode_tokens(processor, token_ids: torch.Tensor, num_tokens: int = 20):
        """Decode and print first N tokens."""
        print(f"\n🔤 FIRST {num_tokens} TOKENS:")
        
        if token_ids.dim() > 1:
            token_ids = token_ids[0]  # Take first batch
        
        tokens_to_show = token_ids[:num_tokens].tolist()
        decoded = processor.tokenizer.convert_ids_to_tokens(tokens_to_show)
        
        for i, (token_id, token) in enumerate(zip(tokens_to_show, decoded)):
            print(f"  {i:3d}: {token_id:6d} -> '{token}'")
        
        print()


class GenerationTracer:
    """Trace generation process step by step."""
    
    def __init__(self):
        self.generation_steps = []
    
    def trace_step(self, step: int, input_ids: torch.Tensor, 
                   logits: torch.Tensor, next_token: int):
        """Record information about a generation step."""
        # Get top-k predictions
        topk_logits, topk_ids = torch.topk(logits[0, -1], k=5)
        
        self.generation_steps.append({
            'step': step,
            'sequence_length': input_ids.shape[1],
            'next_token': next_token,
            'top5_tokens': topk_ids.tolist(),
            'top5_logits': topk_logits.tolist(),
        })
    
    def print_trace(self, processor):
        """Print generation trace."""
        print(f"\n{'='*70}")
        print("🎯 GENERATION TRACE")
        print(f"{'='*70}\n")
        
        for step_info in self.generation_steps:
            step = step_info['step']
            next_token = step_info['next_token']
            
            # Decode token
            decoded = processor.tokenizer.decode([next_token])
            
            print(f"Step {step:3d}: Generated token {next_token:6d} -> '{decoded}'")
            print(f"  Sequence length: {step_info['sequence_length']}")
            print(f"  Top-5 predictions:")
            
            for rank, (tok_id, logit) in enumerate(zip(step_info['top5_tokens'], 
                                                       step_info['top5_logits']), 1):
                tok_str = processor.tokenizer.decode([tok_id])
                print(f"    {rank}. {tok_id:6d} ({logit:8.4f}) -> '{tok_str}'")
            print()
        
        print(f"{'='*70}\n")


# Utility functions for quick debugging
def quick_inspect(name: str, tensor: torch.Tensor):
    """Quick tensor inspection (one-liner)."""
    TensorInspector.inspect(name, tensor, detailed=True)


def print_shapes(inputs: Dict[str, Any], title: str = "Tensor Shapes"):
    """Print shapes of all tensors in a dictionary."""
    print(f"\n{'='*70}")
    print(f"📐 {title}")
    print(f"{'='*70}")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: {tuple(value.shape)}")
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                print(f"  {key:25s}: List of {len(value)} tensors, first shape: {tuple(value[0].shape)}")
    
    print(f"{'='*70}\n")


def monitor_memory():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        
        print(f"\n💾 GPU Memory Usage:")
        print(f"  Allocated:     {allocated:.2f} GB")
        print(f"  Reserved:      {reserved:.2f} GB")
        print(f"  Peak allocated: {max_allocated:.2f} GB")
        print()
    else:
        print("\n⚠️  CUDA not available\n")


if __name__ == "__main__":
    print("🛠️  Qwen3VL Debugging Helpers Loaded")
    print("\nAvailable utilities:")
    print("  - TensorInspector: Detailed tensor inspection")
    print("  - ForwardHookManager: Hook into model layers")
    print("  - ModelArchitectureInspector: Print model structure")
    print("  - ProcessorDebugger: Debug input processing")
    print("  - GenerationTracer: Trace generation step-by-step")
    print("\nQuick functions:")
    print("  - quick_inspect(name, tensor): Quick tensor inspection")
    print("  - print_shapes(dict): Print all tensor shapes")
    print("  - monitor_memory(): Check GPU memory usage")

