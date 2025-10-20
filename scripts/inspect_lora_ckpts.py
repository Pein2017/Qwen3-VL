#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration


def list_files(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


def load_model(checkpoint_path: str):
    # Load on CPU to avoid OOM; we only inspect names
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        device_map='cpu',
    )
    return model


def find_lora_and_modules_to_save(model) -> Tuple[List[str], List[str]]:
    lora_param_names: List[str] = []
    mts_param_names: List[str] = []
    for name, _ in model.named_parameters():
        lname = name.lower()
        if 'lora_' in lname:
            lora_param_names.append(name)
        if '.modules_to_save.' in name:
            mts_param_names.append(name)
    return lora_param_names, mts_param_names


def split_llm_vs_vision(names: List[str]) -> Tuple[List[str], List[str]]:
    vision_markers = ('vision', 'visual')
    vision = [n for n in names if any(m in n.lower() for m in vision_markers)]
    llm = [n for n in names if n not in vision]
    return llm, vision


def summarize_checkpoint(path: str) -> Dict:
    files = list_files(path)
    # Adapter-style layout flags
    has_adapter_cfg = os.path.exists(os.path.join(path, 'adapter_config.json'))
    has_default_adapter_cfg = os.path.exists(os.path.join(path, 'default', 'adapter_config.json'))

    try:
        model = load_model(path)
    except Exception as e:
        return {
            'path': path,
            'files': files,
            'error': f'Failed to load model: {repr(e)}',
        }

    lora_params, mts_params = find_lora_and_modules_to_save(model)
    lora_llm, lora_vision = split_llm_vs_vision(lora_params)
    mts_llm, mts_vision = split_llm_vs_vision(mts_params)

    result = {
        'path': path,
        'files': files,
        'has_adapter_config': has_adapter_cfg,
        'has_default_adapter_config': has_default_adapter_cfg,
        'num_parameters_total': sum(p.numel() for _, p in model.named_parameters()),
        'lora': {
            'total': len(lora_params),
            'llm': len(lora_llm),
            'vision': len(lora_vision),
            'sample_first_20': lora_params[:20],
        },
        'modules_to_save': {
            'total': len(mts_params),
            'llm': len(mts_llm),
            'vision': len(mts_vision),
            'sample_first_20': mts_params[:20],
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', required=True, help='One or more checkpoint directories')
    args = parser.parse_args()

    outputs = []
    for p in args.paths:
        print(f'\n==== Inspecting: {p} ====')
        if not os.path.isdir(p):
            print(f'[ERROR] Not a directory: {p}')
            continue
        res = summarize_checkpoint(p)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        outputs.append(res)

    # Basic comparison if two paths provided
    if len(outputs) == 2:
        a, b = outputs
        print('\n==== Comparison (LoRA counts) ====')
        print(json.dumps({
            a['path']: a.get('lora', {}),
            b['path']: b.get('lora', {}),
        }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    # Keep CUDA off to reduce memory usage for inspection
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    main()


