"""
Enhanced Qwen3VL Demo with Comprehensive Debugging
This demo shows how to use the debugging helpers to inspect model internals.
"""

import os
import sys
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Import debugging helpers
from debug_helpers import (
    TensorInspector,
    ForwardHookManager,
    ModelArchitectureInspector,
    ProcessorDebugger,
    quick_inspect,
    print_shapes,
    monitor_memory,
)


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(img)
    return images


def main() -> None:
    # Configuration
    checkpoint_path = "model_cache/models/Qwen/Qwen3-VL-4B-Instruct"
    image_paths = [
        'demo/images/QC-20230106-0000211_16517.jpeg',
        'demo/images/QC-20230106-0000211_16519.jpeg',
    ]
    prompt = "简要描述这（些）图片。请输出物体的具体坐标。"
    
    # Debugging options
    ENABLE_HOOKS = True  # Set to False to disable hook-based debugging
    INSPECT_ARCHITECTURE = True
    INSPECT_PROCESSOR_OUTPUT = True
    INSPECT_GENERATION_INPUT = True
    
    if not image_paths:
        raise ValueError("Please set at least one path in image_paths.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}\n")
    
    # ========================================================================
    # PHASE 1: Load Processor
    # ========================================================================
    print("="*80)
    print("PHASE 1: Loading Processor")
    print("="*80)
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    print(f"✅ Processor loaded: {processor.__class__.__name__}")
    print(f"   Image token: '{processor.image_token}' (ID: {processor.image_token_id})")
    print(f"   Video token: '{processor.video_token}' (ID: {processor.video_token_id})")
    print(f"   Vision start: '{processor.vision_start_token}' (ID: {processor.vision_start_token_id})")
    print(f"   Vision end: '{processor.vision_end_token}' (ID: {processor.vision_end_token_id})")
    
    monitor_memory()
    
    # ========================================================================
    # PHASE 2: Load Model
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Loading Model")
    print("="*80)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path, 
        torch_dtype="auto"
    )
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model.__class__.__name__}")
    
    monitor_memory()
    
    # Inspect model architecture
    if INSPECT_ARCHITECTURE:
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE INSPECTION")
        print("="*80)
        
        ModelArchitectureInspector.count_parameters(model)
        
        print("\n📊 Model Components:")
        print(f"  Vision model: {model.visual.__class__.__name__}")
        print(f"  Language model: {model.language_model.__class__.__name__}")
        print(f"  LM head: {model.lm_head.__class__.__name__}")
        
        # Print architecture (limited depth to avoid clutter)
        ModelArchitectureInspector.print_architecture(model, max_depth=2)
    
    # ========================================================================
    # PHASE 3: Setup Debugging Hooks
    # ========================================================================
    if ENABLE_HOOKS:
        print("\n" + "="*80)
        print("PHASE 3: Setting Up Debugging Hooks")
        print("="*80)
        
        hook_manager = ForwardHookManager()
        
        # Register hooks on key modules
        modules_to_hook = [
            'visual.patch_embed',          # Patch embedding
            'visual.blocks.0',             # First vision transformer block
            'visual.blocks.-1',            # Last vision transformer block (won't match, just example)
            'visual.merger',               # Patch merger
            'language_model.layers.0',     # First language model layer
            'language_model.layers.1',     # Second language model layer
            'lm_head',                     # Final projection
        ]
        
        hook_manager.register_module_hooks(model, modules_to_hook)
        print()
    
    # ========================================================================
    # PHASE 4: Load and Process Images
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Loading and Processing Images")
    print("="*80)
    
    print(f"Loading {len(image_paths)} images...")
    images = load_images(image_paths)
    
    for i, img in enumerate(images):
        print(f"  Image {i}: Size={img.size}, Mode={img.mode}")
    
    # Build messages
    message_content = [{"type": "image", "image": img} for img in images]
    message_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": message_content}]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print(f"\n📝 Chat template applied:")
    print(f"  Text length: {len(text)} characters")
    print(f"  First 200 chars: {text[:200]}...")
    
    # Process inputs
    print("\n🔄 Processing inputs...")
    inputs = processor(
        images=images,
        text=text,
        return_tensors="pt",
    )
    
    # Inspect processor output
    if INSPECT_PROCESSOR_OUTPUT:
        ProcessorDebugger.inspect_processor_output(inputs)
        
        # Decode first few tokens
        ProcessorDebugger.decode_tokens(processor, inputs['input_ids'], num_tokens=30)
    
    # Move to device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    
    print_shapes(inputs, "Input Shapes After Moving to Device")
    
    # ========================================================================
    # PHASE 5: Detailed Input Inspection
    # ========================================================================
    if INSPECT_GENERATION_INPUT:
        print("\n" + "="*80)
        print("PHASE 5: Detailed Input Tensor Inspection")
        print("="*80)
        
        inspector = TensorInspector()
        
        if 'input_ids' in inputs:
            inspector.inspect("input_ids", inputs['input_ids'], detailed=False)
        
        if 'attention_mask' in inputs:
            inspector.inspect("attention_mask", inputs['attention_mask'], detailed=False)
        
        if 'pixel_values' in inputs:
            inspector.inspect("pixel_values (images)", inputs['pixel_values'], detailed=True)
        
        if 'pixel_values_videos' in inputs:
            inspector.inspect("pixel_values_videos", inputs['pixel_values_videos'], detailed=True)
        
        if 'image_grid_thw' in inputs:
            inspector.inspect("image_grid_thw", inputs['image_grid_thw'], detailed=False)
            print("\n📊 Image Grid Breakdown:")
            for i, grid in enumerate(inputs['image_grid_thw']):
                t, h, w = grid.tolist()
                print(f"  Image {i}: Temporal={t}, Height_grids={h}, Width_grids={w}")
                print(f"           Total patches: {t * h * w}")
        
        if 'video_grid_thw' in inputs:
            inspector.inspect("video_grid_thw", inputs['video_grid_thw'], detailed=False)
    
    monitor_memory()
    
    # ========================================================================
    # PHASE 6: Generation
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: Running Generation")
    print("="*80)
    
    print("🔮 Generating response...")
    print(f"  Max new tokens: 2048")
    print(f"  Sampling: Greedy (do_sample=False)")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
    
    print(f"✅ Generation complete!")
    print(f"  Generated sequence length: {generated_ids.shape[1]}")
    print(f"  Original input length: {inputs['input_ids'].shape[1]}")
    print(f"  New tokens generated: {generated_ids.shape[1] - inputs['input_ids'].shape[1]}")
    
    monitor_memory()
    
    # ========================================================================
    # PHASE 7: Inspect Activations (if hooks enabled)
    # ========================================================================
    if ENABLE_HOOKS:
        print("\n" + "="*80)
        print("PHASE 7: Activation Inspection")
        print("="*80)
        
        hook_manager.print_summary()
        
        # Save activation summary
        hook_manager.save_activations("activation_summary.json")
        
        # Clean up hooks
        hook_manager.remove_hooks()
        print("✅ Hooks removed")
    
    # ========================================================================
    # PHASE 8: Decode Output
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 8: Decoding Output")
    print("="*80)
    
    # Trim input prefix
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    print(f"🔤 Decoding {len(generated_ids_trimmed[0])} new tokens...")
    
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    
    print("\n" + "="*80)
    print("📄 MODEL RESPONSE")
    print("="*80)
    print(outputs[0] if outputs else "")
    print("="*80)
    
    # ========================================================================
    # PHASE 9: Final Statistics
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 9: Final Statistics")
    print("="*80)
    
    monitor_memory()
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Tips for debugging:")
    print("  1. Set breakpoints in transformers library files:")
    print("     - modeling_qwen3_vl.py:1315 (forward)")
    print("     - modeling_qwen3_vl.py:703 (vision forward)")
    print("  2. Enable more hooks by adding module names to modules_to_hook")
    print("  3. Use TensorInspector to examine any intermediate tensors")
    print("  4. Check activation_summary.json for layer-wise information")
    print("\n📚 See QWEN3VL_DEPTH_FIRST_ANALYSIS.md for detailed guide")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # Show memory even on error
        monitor_memory()

