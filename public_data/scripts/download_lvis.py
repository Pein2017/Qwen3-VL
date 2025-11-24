#!/usr/bin/env python3
"""
Download LVIS v1.0 annotations and COCO 2017 images.

LVIS uses COCO 2017 images, so we download both:
- LVIS annotations (train + val)
- COCO 2017 images (train + val)

Total download: ~25 GB
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Download URLs
LVIS_URLS = {
    "train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
}

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
}

# Expected file sizes (approximate, in MB)
EXPECTED_SIZES = {
    "lvis_v1_train.json.zip": 50,
    "lvis_v1_val.json.zip": 10,
    "train2017.zip": 18000,
    "val2017.zip": 1000,
}


def download_file(url: str, output_path: str, expected_size_mb: Optional[int] = None) -> bool:
    """
    Download file using wget with progress bar.
    
    Args:
        url: Download URL
        output_path: Local output path
        expected_size_mb: Expected file size in MB (for validation)
        
    Returns:
        True if successful
        
    Raises:
        RuntimeError: If download fails
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {os.path.basename(output_path)}")
    print(f"From: {url}")
    print(f"To: {output_path}")
    print(f"{'='*60}")
    
    # Check if already downloaded
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ File already exists ({size_mb:.1f} MB)")
        
        if expected_size_mb and abs(size_mb - expected_size_mb) > expected_size_mb * 0.1:
            print(f"  ! Warning: Size mismatch (expected ~{expected_size_mb} MB)")
            response = input("  Re-download? [y/N]: ").strip().lower()
            if response != 'y':
                return True
        else:
            return True
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with wget (resume support)
    try:
        cmd = [
            "wget",
            "--continue",  # Resume if interrupted
            "--progress=bar:force",
            "--show-progress",
            url,
            "-O", output_path
        ]
        
        result = subprocess.run(cmd, check=True)
        
        # Verify download
        if not os.path.exists(output_path):
            raise RuntimeError("Download completed but file not found")
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")
        return True
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e}")
    except Exception as e:
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Download error: {e}")


def unzip_file(zip_path: str, output_dir: str, *, remove_zip: bool = False) -> None:
    """
    Unzip file to output directory.
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract to
        remove_zip: Remove zip after extraction
        
    Raises:
        RuntimeError: If unzip fails
    """
    print(f"\nExtracting: {os.path.basename(zip_path)}")
    print(f"To: {output_dir}")
    
    # Check if already extracted
    expected_name = os.path.basename(zip_path).replace(".zip", "")
    if os.path.exists(os.path.join(output_dir, expected_name)):
        print(f"✓ Already extracted: {expected_name}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cmd = ["unzip", "-q", zip_path, "-d", output_dir]
        subprocess.run(cmd, check=True)
        print(f"✓ Extracted successfully")
        
        if remove_zip:
            os.remove(zip_path)
            print(f"✓ Removed zip file")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Extraction failed: {e}")


def download_lvis_annotations(output_dir: str, splits: list = ["train", "val"]) -> None:
    """Download LVIS annotation files."""
    print("\n" + "="*60)
    print("STEP 1: Download LVIS Annotations")
    print("="*60)
    
    ann_dir = os.path.join(output_dir, "lvis", "raw", "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    
    for split in splits:
        if split not in LVIS_URLS:
            print(f"! Unknown split: {split}")
            continue
        
        url = LVIS_URLS[split]
        filename = f"lvis_v1_{split}.json.zip"
        zip_path = os.path.join(ann_dir, filename)
        
        download_file(url, zip_path, EXPECTED_SIZES.get(filename))
        unzip_file(zip_path, ann_dir, remove_zip=False)


def download_coco_images(output_dir: str, splits: list = ["train", "val"]) -> None:
    """Download COCO 2017 images."""
    print("\n" + "="*60)
    print("STEP 2: Download COCO 2017 Images")
    print("="*60)
    print("Note: This may take 1-2 hours depending on connection")
    
    img_dir = os.path.join(output_dir, "lvis", "raw", "images")
    os.makedirs(img_dir, exist_ok=True)
    
    for split in splits:
        url_key = f"{split}_images"
        if url_key not in COCO_URLS:
            print(f"! Unknown split: {split}")
            continue
        
        url = COCO_URLS[url_key]
        filename = f"{split}2017.zip"
        zip_path = os.path.join(img_dir, filename)
        
        download_file(url, zip_path, EXPECTED_SIZES.get(filename))
        unzip_file(zip_path, img_dir, remove_zip=False)


def verify_download(output_dir: str) -> None:
    """Verify all required files exist."""
    print("\n" + "="*60)
    print("STEP 3: Verify Download")
    print("="*60)
    
    lvis_dir = os.path.join(output_dir, "lvis", "raw")
    
    required_files = [
        "annotations/lvis_v1_train.json",
        "annotations/lvis_v1_val.json",
        "images/train2017",
        "images/val2017"
    ]
    
    all_ok = True
    for rel_path in required_files:
        full_path = os.path.join(lvis_dir, rel_path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {rel_path}")
        
        if exists and os.path.isdir(full_path):
            count = len(os.listdir(full_path))
            print(f"  ({count} files)")
        elif exists:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ({size_mb:.1f} MB)")
        
        all_ok = all_ok and exists
    
    if all_ok:
        print("\n✓ All files downloaded successfully!")
    else:
        print("\n✗ Some files are missing. Please re-run the script.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download LVIS v1.0 dataset (annotations + COCO images)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data to /data/public_data
  python download_lvis.py
  
  # Download only train split
  python download_lvis.py --splits train
  
  # Custom output directory
  python download_lvis.py --output_dir /path/to/data

Storage Requirements:
  - LVIS annotations: ~60 MB
  - COCO train images: ~18 GB
  - COCO val images: ~1 GB
  - Total: ~25 GB
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/public_data",
        help="Output directory (default: /data/public_data)"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Splits to download (default: train val)"
    )
    
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip downloading COCO images (only download annotations)"
    )
    
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir)
    
    print("="*60)
    print("LVIS v1.0 Dataset Downloader")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Skip images: {args.skip_images}")
    print("="*60)
    
    # Check dependencies
    for cmd in ["wget", "unzip"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            print(f"✗ Error: '{cmd}' not found. Please install it first.")
            sys.exit(1)
    
    try:
        # Download annotations
        download_lvis_annotations(output_dir, args.splits)
        
        # Download images
        if not args.skip_images:
            download_coco_images(output_dir, args.splits)
        else:
            print("\n! Skipped downloading COCO images (--skip_images)")
        
        # Verify
        if not args.skip_images:
            verify_download(output_dir)
        
        print("\n" + "="*60)
        print("✓ Download Complete!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"  1. Convert to JSONL:")
        print(f"     python scripts/convert_lvis.py")
        print(f"  2. Create samples:")
        print(f"     python scripts/sample_dataset.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

