"""SFT runner - pure YAML config-driven, no CLI arguments for hyperparameters"""
import argparse
import copy
import os
import re

from swift.llm.train.sft import SwiftSft
from swift.trainers import TrainerFactory

from .datasets import DenseCaptionDataset
from .datasets.augment import AugmentationConfig
from .config import ConfigLoader

# Use the model's native chat_template (JSON/Jinja) shipped with the tokenizer

def parse_args():
    """Parse minimal runtime arguments.
    
    All training configuration comes from YAML files.
    CLI only accepts runtime settings like config path and debug mode.
    """
    parser = argparse.ArgumentParser(
        description="SFT training with YAML configuration - zero CLI hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with config
  python -m src.sft --config configs/qwen3vl_lora.yaml
  
  # With inheritance from base config
  python -m src.sft --config configs/qwen3vl_lora.yaml --base_config configs/base.yaml
  
  # Debug mode
  python -m src.sft --config configs/debug.yaml --debug
        """
    )
    
    # Required: config file path
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (required)'
    )
    
    # Optional: base config for inheritance
    parser.add_argument(
        '--base_config',
        type=str,
        default=None,
        help='Path to base YAML config for inheritance (optional)'
    )
    
    # Runtime: debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging and print full config'
    )
    
    return parser.parse_args()


def main():
    """Main training entry point - pure config-driven."""
    args = parse_args()
    
    print("=" * 70)
    print("  MS-Swift Training with YAML Configuration")
    print("=" * 70)
    print(f"[INFO] Config file: {args.config}")
    if args.base_config:
        print(f"[INFO] Base config: {args.base_config}")
    print("=" * 70)
    
    # Load configuration from YAML
    print(f"\n[INFO] Loading configuration...")
    train_args, custom_config = ConfigLoader.load_training_config(
        args.config,
        args.base_config
    )
    # Append run_name to output_dir and logging_dir to form final paths
    try:
        run_name = getattr(train_args, 'run_name', None)
        training_args = getattr(train_args, 'training_args', None)

        # Resolve and update output_dir
        base_output_dir = getattr(train_args, 'output_dir', None)
        if base_output_dir is None and training_args is not None:
            base_output_dir = getattr(training_args, 'output_dir', None)
        if run_name and base_output_dir:
            base_output_dir_norm = os.path.normpath(base_output_dir)
            if os.path.basename(base_output_dir_norm) != str(run_name):
                final_output_dir = os.path.join(base_output_dir, str(run_name))
                try:
                    setattr(train_args, 'output_dir', final_output_dir)
                except Exception:
                    pass
                if training_args is not None:
                    try:
                        setattr(training_args, 'output_dir', final_output_dir)
                    except Exception:
                        pass

        # Resolve and update logging_dir (tensorboard dir)
        base_logging_dir = getattr(train_args, 'logging_dir', None)
        if base_logging_dir is None and training_args is not None:
            base_logging_dir = getattr(training_args, 'logging_dir', None)
        if run_name and base_logging_dir:
            base_logging_dir_norm = os.path.normpath(base_logging_dir)
            if os.path.basename(base_logging_dir_norm) != str(run_name):
                final_logging_dir = os.path.join(base_logging_dir, str(run_name))
                try:
                    setattr(train_args, 'logging_dir', final_logging_dir)
                except Exception:
                    pass
                if training_args is not None:
                    try:
                        setattr(training_args, 'logging_dir', final_logging_dir)
                    except Exception:
                        pass
    except Exception:
        # Non-fatal: fall back to directories as provided by YAML
        pass
    
    # Debug mode: print full configuration
    if args.debug:
        print("\n[DEBUG] TrainArguments:")
        for key, value in vars(train_args).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        print("\n[DEBUG] Custom dataset config:")
        for key, value in custom_config.items():
            print(f"  {key}: {value}")
        print("\n" + "=" * 70 + "\n")
    
    # Auto-configure ROOT_IMAGE_DIR from JSONL directory
    train_jsonl = custom_config.get('train_jsonl') or custom_config.get('jsonl')
    if not train_jsonl:
        raise ValueError("Config must specify 'custom.train_jsonl' or 'custom.jsonl'")
    
    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        root_dir = os.path.abspath(os.path.dirname(train_jsonl))
        os.environ["ROOT_IMAGE_DIR"] = root_dir
        print(f"[INFO] Set ROOT_IMAGE_DIR={root_dir}")
    
    # Initialize SwiftSft with TrainArguments object directly
    print("[INFO] Initializing ms-swift SFT...")
    sft = SwiftSft(train_args)
    print(f"[INFO] Model: {train_args.model}")
    print(f"[INFO] Training type: {train_args.train_type}")
    if train_args.train_type == 'lora':
        print(f"[INFO] LoRA rank: {train_args.lora_rank}, alpha: {train_args.lora_alpha}")
    
    # Configure augmentation if requested
    augmenter = None
    augment_prob = custom_config.get('augment_prob', 0.0)
    if augment_prob > 0:
        augmenter = AugmentationConfig(prob=augment_prob)
        print(f"[INFO] Augmentation enabled: prob={augment_prob}")
    
    # Sample limits for quick smoke tests
    shared_sample_limit = custom_config.get('sample_limit')
    train_sample_limit = custom_config.get('train_sample_limit', shared_sample_limit)
    val_sample_limit = custom_config.get('val_sample_limit', shared_sample_limit)
    if train_sample_limit:
        print(f"[INFO] Train sample limit: {train_sample_limit}")
    if val_sample_limit:
        print(f"[INFO] Val sample limit: {val_sample_limit}")
    
    # Build training dataset
    print(f"\n[INFO] Loading training dataset: {train_jsonl}")
    # Require minimal explicit keys; others have sane defaults
    required_keys = [
        'user_prompt', 'emit_norm'
    ]
    missing = [k for k in required_keys if k not in custom_config]
    if missing:
        raise ValueError(f"Missing required custom.* keys: {missing}")

    # Build dynamic pairing config
    from .datasets.dynamic_pair import DynamicPairingConfig
    dp_config = DynamicPairingConfig(
        images_per_user_turn=int(custom_config.get('images_per_user_turn', 2)),
        pre_tokenize=False,
        seed=42,
    )

    dataset = DenseCaptionDataset.from_jsonl(
        train_jsonl,
        template=sft.template,
        user_prompt=custom_config['user_prompt'],
        emit_norm=custom_config['emit_norm'],
        config=dp_config,
        augmenter=augmenter,
        sample_limit=train_sample_limit,
    )
    print(f"[INFO] Training dataset size: {len(dataset)}")

    # Optional: dump conversation text-only (no tokens, no images) and full tokens
    dump_conv = bool(custom_config.get('dump_conversation_text', False) or args.debug)
    if dump_conv and len(dataset) > 0:
        try:
            template = dataset.template
            template.set_mode('pt')
            try:
                sample_encoded = dataset[0]
            finally:
                template.set_mode('train')

            input_ids = sample_encoded.get('input_ids') if isinstance(sample_encoded, dict) else None
            if input_ids is None:
                raise ValueError('Sample does not contain input_ids for dumping conversation text.')
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()

            raw_text = template.tokenizer.decode(input_ids, skip_special_tokens=False)

            def _compress_image_pad(text: str) -> str:
                def repl(match: re.Match) -> str:
                    count = match.group(0).count('<|image_pad|>')
                    return f'<|image_pad|>*{count}'

                return re.sub(r'(?:<\|image_pad\|>)+', repl, text)

            raw_text = _compress_image_pad(raw_text)

            assistant_gt = None
            try:
                rec_a = copy.deepcopy(dataset.base_records[0])
                rec_b = copy.deepcopy(dataset.base_records[0])
                merged = dataset.pair_message_builder(rec_a, rec_b)
                assistant_turn = next(
                    (turn for turn in merged.get('messages', []) if turn.get('role') == 'assistant'),
                    None,
                )
                if assistant_turn:
                    contents = assistant_turn.get('content') or []
                    for item in contents:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            assistant_gt = item.get('text')
                            break
            except Exception as inner_e:
                print(f"[WARNING] Failed to extract assistant GT: {inner_e}")

            print("\n[DEBUG] Conversation (raw):\n" + raw_text + "\n")
            if assistant_gt:
                print("[DEBUG] Assistant GT:\n" + assistant_gt + "\n")

            dump_path = custom_config.get('dump_conversation_path') or 'conversation_text.txt'
            if not os.path.isabs(dump_path):
                dump_path = os.path.join(train_args.output_dir, dump_path)
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
                if not raw_text.endswith('\n'):
                    f.write('\n')
                if assistant_gt:
                    if not raw_text.endswith('\n'):
                        f.write('\n')
                    f.write("\n--- Assistant GT ---\n")
                    f.write(assistant_gt)
                    if not assistant_gt.endswith('\n'):
                        f.write('\n')
            print(f"[INFO] Conversation text saved to: {dump_path}")
        except Exception as e:
            print(f"[WARNING] Failed to dump conversation text: {e}")
    
    # Build validation dataset if provided
    eval_dataset = None
    val_jsonl = custom_config.get('val_jsonl')
    if val_jsonl:
        print(f"[INFO] Loading validation dataset: {val_jsonl}")
        eval_dataset = DenseCaptionDataset.from_jsonl(
            val_jsonl,
            template=sft.template,
            user_prompt=custom_config['user_prompt'],
            emit_norm=custom_config['emit_norm'],
            config=dp_config,
            augmenter=None,  # No augmentation for validation
            sample_limit=val_sample_limit,
        )
        print(f"[INFO] Validation dataset size: {len(eval_dataset)}")
    
    # Sample printing disabled to avoid dumping labels/ids
    
    # Setup trainer
    print("\n[INFO] Setting up trainer...")
    data_collator = sft._get_data_collator()
    trainer_cls = TrainerFactory.get_trainer_cls(train_args)
    
    trainer = trainer_cls(
        model=sft.model,
        args=train_args.training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=sft.callbacks,
        template=sft.template,
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("  Starting Training")
    print("=" * 70)
    print(f"  Output directory: {train_args.output_dir}")
    print(f"  Epochs: {train_args.num_train_epochs}")
    print(f"  Effective batch size: {train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps}")
    print("=" * 70 + "\n")
    
    sft.train(trainer)


if __name__ == "__main__":
    main()



