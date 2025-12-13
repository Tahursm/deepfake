"""
Script to prepare dataset structure and metadata.
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse


def create_metadata(
    data_dir: str,
    split: str,
    real_dir: str = None,
    fake_dir: str = None
) -> List[Dict]:
    """
    Create metadata JSON for dataset split.
    
    Args:
        data_dir: Root data directory
        split: Split name ('train', 'val', 'test')
        real_dir: Directory containing real videos (relative to data_dir/split)
        fake_dir: Directory containing fake videos (relative to data_dir/split)
        
    Returns:
        List of sample dictionaries
    """
    data_path = Path(data_dir)
    split_path = data_path / split
    
    samples = []
    
    # Default directory structure: data_dir/split/{real|fake}/
    if real_dir is None:
        real_dir = 'real'
    if fake_dir is None:
        fake_dir = 'fake'
    
    real_path = split_path / real_dir
    fake_path = split_path / fake_dir
    
    # Add real samples
    if real_path.exists():
        for video_file in real_path.glob('*.mp4'):
            samples.append({
                'video_path': str(video_file),
                'label': 0  # 0 = real
            })
    
    # Add fake samples
    if fake_path.exists():
        for video_file in fake_path.glob('*.mp4'):
            samples.append({
                'video_path': str(video_file),
                'label': 1  # 1 = fake
            })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset metadata')
    parser.add_argument('--data_dir', type=str, required=True, help='Root data directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for split in args.splits:
        print(f"Processing {split} split...")
        samples = create_metadata(args.data_dir, split)
        
        if len(samples) == 0:
            print(f"Warning: No samples found for {split} split")
            continue
        
        # Save metadata
        metadata_file = data_path / f'{split}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        # Count labels
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"  Created metadata for {len(samples)} samples")
        print(f"  Real: {real_count}, Fake: {fake_count}")
        print(f"  Saved to: {metadata_file}")
    
    print("\nDataset preparation complete!")


if __name__ == '__main__':
    main()

