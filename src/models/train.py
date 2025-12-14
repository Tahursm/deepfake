"""
Training script for deepfake detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import argparse
import sys
import platform
import os

# Handle imports for both direct execution and module import
try:
    from .complete_model import DeepfakeDetectionModel
    from ..utils.dataset import DeepfakeDataset
except ImportError:
    # If relative imports fail, use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.complete_model import DeepfakeDetectionModel
    from src.utils.dataset import DeepfakeDataset

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance with per-class alpha weights."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, alpha_per_class: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # alpha_per_class: tensor of shape [num_classes] for per-class weighting
        # If None, uses uniform alpha. If provided, overrides self.alpha
        if alpha_per_class is not None:
            self.register_buffer('alpha_per_class', alpha_per_class)
        else:
            self.alpha_per_class = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply per-class alpha if provided
        if self.alpha_per_class is not None:
            # Get alpha for each sample based on its class
            alpha_t = self.alpha_per_class[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class Trainer:
    """Trainer class for deepfake detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: Path
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate class weights for imbalanced datasets
        # Count samples per class in training set
        print("📊 Calculating class weights from training data...")
        class_counts = torch.zeros(2)  # 2 classes: real (0) and fake (1)
        for batch in train_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label.item()] += 1
        
        # Calculate inverse frequency weights (more weight to minority class)
        total_samples = class_counts.sum()
        if total_samples > 0 and class_counts.min() > 0:
            # Inverse frequency weighting: weight = total / (num_classes * count)
            class_weights = total_samples / (2.0 * class_counts)
            # Normalize so they sum to num_classes
            class_weights = class_weights / class_weights.sum() * 2.0
            print(f"   Class distribution - Real: {class_counts[0]:.0f}, Fake: {class_counts[1]:.0f}")
            print(f"   Class weights - Real: {class_weights[0]:.3f}, Fake: {class_weights[1]:.3f}")
        else:
            class_weights = torch.ones(2)  # Equal weights if can't calculate
            print("⚠️  Could not calculate class weights, using equal weights")
        
        # Loss function
        if config.get('use_focal_loss', False):
            # Use per-class alpha weights based on class imbalance
            # Give more weight to fake class (minority) if it's underrepresented
            focal_alpha_per_class = config.get('focal_alpha_per_class', None)
            if focal_alpha_per_class is None:
                # Auto-calculate: use class weights as alpha (inverse frequency)
                # This gives more weight to the minority class (fake)
                focal_alpha_per_class = class_weights.to(device)
            else:
                focal_alpha_per_class = torch.tensor(focal_alpha_per_class, dtype=torch.float32, device=device)
            
            self.criterion = FocalLoss(
                alpha=config.get('focal_alpha', 1.0),
                gamma=config.get('focal_gamma', 2.0),
                alpha_per_class=focal_alpha_per_class
            )
            print(f"✅ Focal loss with per-class alpha: Real={focal_alpha_per_class[0]:.3f}, Fake={focal_alpha_per_class[1]:.3f}")
        else:
            # Use weighted CrossEntropyLoss for class imbalance
            weight = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=config.get('label_smoothing', 0.0)
            )
            print(f"✅ Weighted CrossEntropyLoss with class weights: Real={weight[0]:.3f}, Fake={weight[1]:.3f}")
        
        self.auxiliary_loss_weight = config.get('auxiliary_loss_weight', 0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2)
        )
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100),
                eta_min=config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.get('learning_rate', 1e-4),
                epochs=config.get('epochs', 100),
                steps_per_epoch=len(train_loader)
            )
        else:
            self.scheduler = None
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(checkpoint_dir / 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.train_losses = []
        self.val_metrics = []
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            mouth_frames = batch['mouth_frames'].to(self.device)
            spectrogram = batch['spectrogram'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                frames=frames,
                spectrogram=spectrogram,
                mouth_frames=mouth_frames,
                return_embeddings=True
            )
            
            # Classification loss
            loss = self.criterion(output['logits'], labels)
            
            # Auxiliary loss (lip-sync)
            if self.model.use_auxiliary_loss:
                aux_loss = self.model.compute_auxiliary_loss(
                    output['lip_embedding'],
                    output['audio_embedding']
                )
                loss = loss + self.auxiliary_loss_weight * aux_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update learning rate (for OneCycleLR)
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                frames = batch['frames'].to(self.device)
                mouth_frames = batch['mouth_frames'].to(self.device)
                spectrogram = batch['spectrogram'].to(self.device)
                labels = batch['label'].to(self.device)
                
                output = self.model(
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
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with AUC: {self.best_val_auc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume: bool = False):
        """Train model."""
        if resume:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Update learning rate (for CosineAnnealingLR)
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
            self.writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 10)
                if epoch >= patience:
                    recent_aucs = [m['auc'] for m in self.val_metrics[-patience:]]
                    if max(recent_aucs) < self.best_val_auc:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        self.writer.close()
        print(f"Training completed. Best Val AUC: {self.best_val_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Windows/Colab compatibility: MediaPipe objects cannot be pickled for multiprocessing
    # Set num_workers=0 on Windows or Colab to avoid pickling issues and segfaults
    is_windows = platform.system() == 'Windows'
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_IP' in os.environ or os.path.exists('/content')
    
    # Force num_workers=0 if config says so, or if on Windows/Colab
    # Note: On Linux with T4 GPU, num_workers can be 4-8 for better performance
    config_num_workers = config['training'].get('num_workers', 4)
    if config_num_workers == 0 or is_windows or is_colab:
        num_workers = 0
        if is_windows:
            print("Windows detected: Setting num_workers=0 to avoid multiprocessing pickling issues")
        elif is_colab:
            print("Colab detected: Setting num_workers=0 to avoid MediaPipe multiprocessing segfault")
    else:
        num_workers = config_num_workers
        print(f"Using num_workers={num_workers} for faster data loading")
    
    # Use pin_memory from config if available, otherwise auto-detect based on CUDA
    pin_memory = config['training'].get('pin_memory', torch.cuda.is_available())
    if pin_memory and torch.cuda.is_available():
        print("pin_memory enabled for faster GPU data transfer")
    
    # Datasets
    train_dataset = DeepfakeDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        num_frames=config['data']['num_frames'],
        augment=True
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        num_frames=config['data']['num_frames'],
        augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Model
    model = DeepfakeDetectionModel(
        video_backbone=config['model']['video_backbone'],
        video_embedding_dim=config['model']['video_embedding_dim'],
        num_frames=config['data']['num_frames'],
        n_mels=config['model']['n_mels'],
        audio_embedding_dim=config['model']['audio_embedding_dim'],
        fusion_dim=config['model']['fusion_dim'],
        num_heads=config['model']['num_heads'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout=config['model']['dropout'],
        use_cross_attention=config['model']['use_cross_attention']
    )
    
    # Trainer
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['epochs'],
        resume=args.resume
    )


if __name__ == '__main__':
    main()

