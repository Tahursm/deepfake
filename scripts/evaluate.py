"""
Evaluation script for trained model.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
import numpy as np
import platform

from src.models.complete_model import DeepfakeDetectionModel
from src.utils.dataset import DeepfakeDataset
from src.utils.metrics import compute_all_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Windows compatibility: MediaPipe objects cannot be pickled for multiprocessing
    is_windows = platform.system() == 'Windows'
    num_workers = 0 if is_windows else 4
    pin_memory = torch.cuda.is_available() and not is_windows
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})
    
    # Dataset
    dataset = DeepfakeDataset(
        data_dir=args.data_dir,
        split=args.split,
        num_frames=config.get('data', {}).get('num_frames', 64),
        augment=False
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Model
    model_config = config.get('model', {})
    model = DeepfakeDetectionModel(
        video_backbone=model_config.get('video_backbone', 'resnet50'),
        video_embedding_dim=model_config.get('video_embedding_dim', 512),
        num_frames=config.get('data', {}).get('num_frames', 64),
        n_mels=model_config.get('n_mels', 128),
        audio_embedding_dim=model_config.get('audio_embedding_dim', 512),
        fusion_dim=model_config.get('fusion_dim', 512)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"Evaluating on {args.split} split...")
    
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(data_loader):
            frames = batch['frames'].to(device)
            mouth_frames = batch['mouth_frames'].to(device)
            spectrogram = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            output = model(
                frames=frames,
                spectrogram=spectrogram,
                mouth_frames=mouth_frames
            )
            
            probs = output['probs']
            preds = torch.argmax(output['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    print_metrics(metrics)
    
    # Save results
    results_file = Path(args.checkpoint).parent / f'evaluation_{args.split}.json'
    import json
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

