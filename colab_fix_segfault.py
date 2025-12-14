"""
QUICK FIX for Segmentation Fault in Colab
==========================================

Run this cell in Colab to fix the DataLoader segmentation fault issue.
This sets num_workers=0 to avoid MediaPipe multiprocessing issues.
"""

import yaml
import os

# Fix 1: Update config file
config_path = 'experiments/configs/default.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set num_workers to 0
    if 'training' in config:
        config['training']['num_workers'] = 0
        print("✅ Updated config: num_workers = 0")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Config file updated!")
    print("   Now re-run the training cell (Cell 6)")
else:
    print("⚠️  Config file not found. Make sure you're in the right directory.")

# Fix 2: The training code will also auto-detect Colab and set num_workers=0
# So even if config has num_workers > 0, it will be overridden

print("\n" + "="*60)
print("FIX APPLIED!")
print("="*60)
print("\nNow re-run Cell 6 (training) and it should work without segfault.")
print("\nNote: Training will be slightly slower with num_workers=0,")
print("but it will be stable and avoid the segmentation fault error.")

