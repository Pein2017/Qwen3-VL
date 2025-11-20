#!/usr/bin/env python3
"""
Simplified Script: Downloading Qwen3-VL Models Using ModelScope (Method 1 Only)

This script downloads Qwen3-VL-4B and Qwen3-VL-8B models from ModelScope
to a local cache directory using the ms-swift framework.

Run with: python download_qwen3vl_example.py
"""

import os
import sys


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)

    dependencies = {
        "swift": "ms-swift",
        "modelscope": "modelscope",
        "transformers": "transformers>=4.57",
        "torch": "torch",
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True


def download_models():
    """
    Download Qwen3-VL 4B and 8B models using Method 1: safe_snapshot_download()

    Models are downloaded to ./model_cache directory
    """
    print("\n" + "=" * 70)
    print("METHOD 1: Using safe_snapshot_download()")
    print("=" * 70)

    try:
        from swift.llm import safe_snapshot_download

        # Set custom cache directory
        cache_dir = os.path.abspath("./model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = cache_dir

        print(f"\nCache directory set to: {cache_dir}")

        # Download 4B model
        print("\n" + "-" * 70)
        print("Downloading Qwen3-VL-4B-Instruct...")
        print("-" * 70)
        model_dir_4b = safe_snapshot_download(
            "Qwen/Qwen3-VL-4B-Instruct",
            use_hf=False,  # Use ModelScope instead of HuggingFace
        )
        print(f"\n✓ 4B model successfully downloaded!")
        print(f"  Location: {model_dir_4b}")

        # Download 8B model
        print("\n" + "-" * 70)
        print("Downloading Qwen3-VL-8B-Instruct...")
        print("-" * 70)
        model_dir_8b = safe_snapshot_download(
            "Qwen/Qwen3-VL-8B-Instruct",
            use_hf=False,  # Use ModelScope instead of HuggingFace
        )
        print(f"\n✓ 8B model successfully downloaded!")
        print(f"  Location: {model_dir_8b}")

        return model_dir_4b, model_dir_8b

    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("  Install ms-swift: pip install ms-swift[framework]")
        return None, None


def main():
    """Main function to download models."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Qwen3-VL Model Downloader (Method 1 Only)  ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Check dependencies first
    if not check_dependencies():
        print("\n⚠ Some dependencies are missing. Please install them and try again.")
        return

    # Download models
    print("\n\nStarting download...\n")
    model_dir_4b, model_dir_8b = download_models()

    # Summary
    if model_dir_4b and model_dir_8b:
        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"""
✓ Successfully downloaded both Qwen3-VL models!

Model Locations:
  4B Model: {model_dir_4b}
  8B Model: {model_dir_8b}

Cache Directory: {os.path.abspath("./model_cache")}

You can now use these models for inference with ms-swift!

Example usage:
  from swift.llm import PtEngine
  
  engine = PtEngine('Qwen/Qwen3-VL-4B-Instruct', use_hf=False)
  response = engine.infer(requests=[{{'query': 'What?', 'images': ['img.jpg']}}])
        """)
    else:
        print("\n✗ Download failed!")


if __name__ == "__main__":
    main()
