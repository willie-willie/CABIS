#!/usr/bin/env python3
"""

@author: Willie
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging

# Add project modules
sys.path.append(str(Path(__file__).parent))

# Import with try-except for better error handling
try:
    from cabis_implementation import ImprovedCABIS, ImprovedHierarchicalCodeEncoder, EnhancedVulnerabilitySynthesizer
    from solidifi_preprocessing import VulnerabilityDataset, prepare_solidifi_data
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure cabis_implementation.py and solidifi_preprocessing.py are in the same directory")
    sys.exit(1)

# Optional import for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Experiment tracking will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom collate function
def custom_collate_fn(batch):
    """Custom collate function that handles mixed data types"""
    # Handle tensor data that can be stacked
    clean_features = torch.stack([item['clean_features'] for item in batch])
    buggy_features = torch.stack([item['buggy_features'] for item in batch])
    vuln_labels = torch.stack([item['vulnerability_labels'] for item in batch])
    
    # Keep string/list data as lists
    clean_code = [item['clean_code'] for item in batch]
    buggy_code = [item['buggy_code'] for item in batch]
    vulnerability_locations = [item.get('vulnerability_locations', []) for item in batch]
    vulnerability_types = [item.get('vulnerability_type', 'unknown') for item in batch]
    source_files = [item.get('source_file', 'unknown') for item in batch]
    
    return {
        'clean_features': clean_features,
        'buggy_features': buggy_features,
        'vulnerability_labels': vuln_labels,
        'clean_code': clean_code,
        'buggy_code': buggy_code,
        'vulnerability_locations': vulnerability_locations,
        'vulnerability_types': vulnerability_types,
        'source_files': source_files
    }

class EnhancedCABISTrainer:
    """Enhanced trainer for CABIS with full dataset support"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize model components
        self.encoder = ImprovedHierarchicalCodeEncoder(self.config).to(self.device)
        self.synthesizer = EnhancedVulnerabilitySynthesizer(self.config).to(self.device)
        
        # Initialize discriminator with spectral normalization for stability
        self.discriminator = self._build_discriminator().to(self.device)
        
        # Initialize analyzer (vulnerability detector)
        self.analyzer = self._build_analyzer().to(self.device)
        
        # Mixed precision training
        self.use_amp = self.config.get('experiment', {}).get('mixed_precision', False)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler(self.device_type)
        else:
            self.use_amp = False  # Disable AMP if CUDA not available
        
        # Get optimizer config with defaults
        optimizer_config = self.config.get('optimizer', {})
        lr = self.config['training']['learning_rate']
        betas = optimizer_config.get('betas', [0.9, 0.999])
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        # Optimizers with different learning rates
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.synthesizer_optimizer = torch.optim.AdamW(
            self.synthesizer.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr * 2,  # Higher LR for discriminator
            betas=betas,
            weight_decay=weight_decay
        )
        
        self.analyzer_optimizer = torch.optim.AdamW(
            self.analyzer.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        # Get scheduler config with defaults
        scheduler_config = self.config.get('scheduler', {})
        T_max = scheduler_config.get('T_max', self.config['training']['epochs'])
        eta_min = scheduler_config.get('eta_min', 1e-5)
        
        # Learning rate schedulers
        self.schedulers = {
            'encoder': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.encoder_optimizer,
                T_max=T_max,
                eta_min=eta_min
            ),
            'synthesizer': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.synthesizer_optimizer,
                T_max=T_max,
                eta_min=eta_min
            ),
            'discriminator': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.discriminator_optimizer,
                T_max=T_max,
                eta_min=eta_min
            ),
            'analyzer': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.analyzer_optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        }
        
        # Loss functions with weights
        self.loss_weights = self.config.get('loss_weights', {
            'adversarial': 1.0,
            'reconstruction': 0.5,
            'classification': 0.3,
            'diversity': 0.2
        })
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.diversity_loss = nn.CosineSimilarity(dim=1)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'diversity_score': 0.0
        }
        
        # Initialize experiment tracking
        self._init_experiment_tracking()
    
    def _build_discriminator(self):
        """Build discriminator with spectral normalization"""
        hidden_dim = self.config['model']['hidden_dim']
        
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _build_analyzer(self):
        """Build multi-class vulnerability analyzer"""
        hidden_dim = self.config['model']['hidden_dim']
        num_classes = len(self.config['vulnerability_types'])
        
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),  # Changed from BatchNorm1d
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _init_experiment_tracking(self):
        """Initialize experiment tracking"""
        logging_config = self.config.get('logging', {})
        
        if logging_config.get('use_wandb', False) and WANDB_AVAILABLE:
            experiment_config = self.config.get('experiment', {})
            wandb.init(
                project="cabis-solidifi-full",
                config=self.config,
                name=f"{experiment_config.get('name', 'cabis')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['full_dataset', '7_vulns']
            )
        
        # Create directories
        paths_config = self.config.get('paths', {})
        for path_key in ['model_dir', 'results_dir', 'log_dir', 'checkpoint_dir']:
            path = paths_config.get(path_key, f'./cabis_project/{path_key}')
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def train(self, train_dataset, val_dataset, num_epochs=None):
        """Enhanced training loop with better strategies"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Create balanced sampler for training
        train_sampler = self._create_balanced_sampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config['training'].get('num_workers', 0),
            pin_memory=self.config['training'].get('pin_memory', False) and torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 0),
            collate_fn=custom_collate_fn
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            logger.info("\n" + "="*60)
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info("="*60)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if (epoch + 1) % self.config['training'].get('validate_every', 2) == 0:
                val_metrics = self._validate_epoch(val_loader)
            else:
                val_metrics = {'total_loss': 0, 'accuracy': 0}
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_every', 5) == 0:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping check
            if self._should_stop_early(val_metrics):
                logger.info("Early stopping triggered")
                break
        
        logger.info("\nTraining completed!")
        self._save_final_model()
    
    def _create_balanced_sampler(self, dataset):
        """Create balanced sampler for vulnerability types"""
        # Get vulnerability type for each sample
        vuln_types = []
        for i in range(len(dataset)):
            vuln_types.append(dataset[i]['vulnerability_type'])
        
        # Count samples per type
        type_counts = defaultdict(int)
        for vt in vuln_types:
            type_counts[vt] += 1
        
        # Create weights
        weights = []
        for vt in vuln_types:
            weights.append(1.0 / type_counts[vt])
        
        return WeightedRandomSampler(weights, len(weights))
    
    def _train_epoch(self, train_loader, epoch):
        """Enhanced training epoch with better strategies"""
        self.encoder.train()
        self.synthesizer.train()
        self.discriminator.train()
        self.analyzer.train()
        
        epoch_metrics = defaultdict(float)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch data
            clean_code = batch['clean_code']
            buggy_code = batch['buggy_code']
            vuln_labels = batch['vulnerability_labels'].to(self.device)
            vuln_types = batch['vulnerability_types']
            
            # Map vulnerability types to indices
            vuln_indices = []
            for vt in vuln_types:
                if vt in self.config['vulnerability_types']:
                    vuln_indices.append(self.config['vulnerability_types'].index(vt))
                else:
                    vuln_indices.append(0)  # Default to first type
            
            # Step 1: Train Discriminator
            d_loss = self._train_discriminator_step(clean_code, buggy_code, vuln_indices)
            epoch_metrics['d_loss'] += d_loss
            
            # Step 2: Train Generator (Encoder + Synthesizer)
            g_loss, g_metrics = self._train_generator_step(clean_code, vuln_indices)
            epoch_metrics['g_loss'] += g_loss
            for k, v in g_metrics.items():
                epoch_metrics[k] += v
            
            # Step 3: Train Analyzer
            a_loss = self._train_analyzer_step(buggy_code, vuln_labels)
            epoch_metrics['a_loss'] += a_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'D': f"{d_loss:.4f}",
                'G': f"{g_loss:.4f}",
                'A': f"{a_loss:.4f}"
            })
            
            # Log batch metrics
            if batch_idx % self.config['logging'].get('log_every_n_steps', 100) == 0:
                self._log_batch_metrics(epoch, batch_idx, epoch_metrics, len(train_loader))
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        return epoch_metrics
    
    def _train_discriminator_step(self, clean_code, buggy_code, vuln_indices):
        """Train discriminator to distinguish real vs generated vulnerabilities"""
        self.discriminator_optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
            # Encode real buggy contracts
            real_embeddings = self.encoder(buggy_code)
            real_validity = self.discriminator(real_embeddings['contract'])
            real_labels = torch.ones(real_validity.shape[0], 1).to(self.device)
            
            # Generate fake vulnerabilities
            with torch.no_grad():
                clean_embeddings = self.encoder(clean_code)
                # Randomly select vulnerability types
                random_vuln_indices = torch.randint(0, len(self.config['vulnerability_types']), (1,)).item()
                fake_embeddings, _ = self.synthesizer(clean_embeddings, random_vuln_indices)
            
            fake_validity = self.discriminator(fake_embeddings.detach())
            fake_labels = torch.zeros(fake_validity.shape[0], 1).to(self.device)
            
            # Discriminator loss
            d_real_loss = self.adversarial_loss(real_validity, real_labels)
            d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.discriminator_optimizer)
            self.scaler.update()
        else:
            d_loss.backward()
            self.discriminator_optimizer.step()
        
        return d_loss.item()
    
    def _train_generator_step(self, clean_code, vuln_indices):
        """Train generator (encoder + synthesizer)"""
        self.encoder_optimizer.zero_grad()
        self.synthesizer_optimizer.zero_grad()
        
        metrics = {}
        
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
            # Encode clean contracts
            clean_embeddings = self.encoder(clean_code)
            
            # Generate vulnerabilities for each type in batch
            total_g_loss = 0
            prev_fake = None  # Initialize prev_fake
            
            for i, vuln_idx in enumerate(vuln_indices[:1]):  # Process one per batch for efficiency
                # Generate fake vulnerability
                fake_embeddings, injection_points = self.synthesizer(clean_embeddings, vuln_idx)
                
                # Adversarial loss - fool discriminator
                fake_validity = self.discriminator(fake_embeddings)
                real_labels = torch.ones(fake_validity.shape[0], 1).to(self.device)
                g_adv_loss = self.adversarial_loss(fake_validity, real_labels)
                
                # Reconstruction loss - preserve functionality
                g_recon_loss = self.reconstruction_loss(
                    fake_embeddings,
                    clean_embeddings['contract']
                )
                
                # Diversity loss - ensure different patterns
                if i > 0 and prev_fake is not None:
                    g_div_loss = 1 - self.diversity_loss(fake_embeddings, prev_fake).mean()
                else:
                    g_div_loss = torch.tensor(0.0).to(self.device)
                
                prev_fake = fake_embeddings.detach()  # Store for next iteration
                
                # Total generator loss with weights
                g_loss = (self.loss_weights['adversarial'] * g_adv_loss + 
                         self.loss_weights['reconstruction'] * g_recon_loss +
                         self.loss_weights['diversity'] * g_div_loss)
                
                total_g_loss += g_loss
            
            total_g_loss /= max(len(vuln_indices[:1]), 1)  # Avoid division by zero
            
            metrics['adv_loss'] = g_adv_loss.item() if 'g_adv_loss' in locals() else 0
            metrics['recon_loss'] = g_recon_loss.item() if 'g_recon_loss' in locals() else 0
            metrics['div_loss'] = g_div_loss.item() if isinstance(g_div_loss, torch.Tensor) else 0
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_g_loss).backward()
            self.scaler.unscale_(self.encoder_optimizer)
            self.scaler.unscale_(self.synthesizer_optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.synthesizer.parameters()),
                self.config['training']['gradient_clip']
            )
            
            self.scaler.step(self.encoder_optimizer)
            self.scaler.step(self.synthesizer_optimizer)
            self.scaler.update()
        else:
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.synthesizer.parameters()),
                self.config['training']['gradient_clip']
            )
            self.encoder_optimizer.step()
            self.synthesizer_optimizer.step()
        
        return total_g_loss.item(), metrics
    
    def _train_analyzer_step(self, buggy_code, vuln_labels):
        """Train analyzer to detect vulnerability types"""
        self.analyzer_optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
            # Encode buggy contracts
            embeddings = self.encoder(buggy_code)
            
            # Predict vulnerability types
            predictions = self.analyzer(embeddings['contract'])
            
            # Classification loss
            a_loss = self.classification_loss(predictions, vuln_labels.argmax(dim=1))
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(a_loss).backward()
            self.scaler.step(self.analyzer_optimizer)
            self.scaler.update()
        else:
            a_loss.backward()
            self.analyzer_optimizer.step()
        
        return a_loss.item()
    
    def _validate_epoch(self, val_loader):
        """Validation epoch with comprehensive metrics"""
        self.encoder.eval()
        self.synthesizer.eval()
        self.analyzer.eval()
        
        val_metrics = defaultdict(float)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                buggy_code = batch['buggy_code']
                vuln_labels = batch['vulnerability_labels'].to(self.device)
                
                # Encode and analyze
                embeddings = self.encoder(buggy_code)
                predictions = self.analyzer(embeddings['contract'])
                
                # Calculate loss
                loss = self.classification_loss(predictions, vuln_labels.argmax(dim=1))
                val_metrics['loss'] += loss.item()
                
                # Store predictions for metrics
                all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
                all_labels.extend(vuln_labels.argmax(dim=1).cpu().numpy())
        
        # Calculate metrics
        val_metrics['loss'] /= len(val_loader)
        val_metrics['accuracy'] = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Per-class accuracy (only if sklearn is available)
        try:
            from sklearn.metrics import classification_report
            
            # FIXED: Add labels parameter to ensure all 7 classes are considered
            report = classification_report(
                all_labels, all_predictions,
                labels=list(range(len(self.config['vulnerability_types']))),  # This is the fix!
                target_names=self.config['vulnerability_types'],
                output_dict=True,
                zero_division=0  # Handle classes with no predictions
            )
            
            val_metrics['per_class_accuracy'] = {
                vuln_type: report[vuln_type]['f1-score']
                for vuln_type in self.config['vulnerability_types']
                if vuln_type in report
            }
        except ImportError:
            logger.warning("sklearn not available, skipping per-class metrics")
            val_metrics['per_class_accuracy'] = {}
        except Exception as e:
            logger.warning(f"Error calculating classification report: {str(e)}")
            val_metrics['per_class_accuracy'] = {}
        
        return {
            'total_loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'per_class': val_metrics['per_class_accuracy']
        }
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to console and tracking systems"""
        # Console logging
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Train - D: {train_metrics['d_loss']:.4f}, "
                   f"G: {train_metrics['g_loss']:.4f}, "
                   f"A: {train_metrics['a_loss']:.4f}")
        
        if val_metrics['total_loss'] > 0:
            logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Log per-class accuracy
            if val_metrics.get('per_class'):
                logger.info("Per-class F1 scores:")
                for vuln_type, score in val_metrics.get('per_class', {}).items():
                    logger.info(f"  {vuln_type}: {score:.4f}")
        
        # Wandb logging
        if self.config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'train/d_loss': train_metrics['d_loss'],
                'train/g_loss': train_metrics['g_loss'],
                'train/a_loss': train_metrics['a_loss'],
                'train/adv_loss': train_metrics.get('adv_loss', 0),
                'train/recon_loss': train_metrics.get('recon_loss', 0),
                'train/div_loss': train_metrics.get('div_loss', 0),
                'val/loss': val_metrics['total_loss'],
                'val/accuracy': val_metrics['accuracy']
            }
            
            # Add per-class metrics
            for vuln_type, score in val_metrics.get('per_class', {}).items():
                log_dict[f'val/f1_{vuln_type}'] = score
            
            wandb.log(log_dict)
    
    def _log_batch_metrics(self, epoch, batch_idx, metrics, total_batches):
        """Log batch-level metrics"""
        if self.config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE:
            step = epoch * total_batches + batch_idx
            wandb.log({
                'batch/d_loss': metrics['d_loss'] / (batch_idx + 1),
                'batch/g_loss': metrics['g_loss'] / (batch_idx + 1),
                'batch/a_loss': metrics['a_loss'] / (batch_idx + 1),
                'step': step
            })
    
    def _save_checkpoint(self, epoch, val_metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state': self.encoder.state_dict(),
            'synthesizer_state': self.synthesizer.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'analyzer_state': self.analyzer.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'synthesizer_optimizer': self.synthesizer_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'analyzer_optimizer': self.analyzer_optimizer.state_dict(),
            'schedulers': {k: v.state_dict() for k, v in self.schedulers.items()},
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save epoch checkpoint
        checkpoint_dir = Path(self.config['paths'].get('checkpoint_dir', './cabis_project/checkpoints'))
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total_loss'] < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = val_metrics['total_loss']
            self.best_metrics['val_accuracy'] = val_metrics['accuracy']
            
            model_dir = Path(self.config['paths'].get('model_dir', './cabis_project/models'))
            best_path = model_dir / 'cabis_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def _save_final_model(self):
        """Save final model"""
        final_checkpoint = {
            'encoder_state': self.encoder.state_dict(),
            'synthesizer_state': self.synthesizer.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'analyzer_state': self.analyzer.state_dict(),
            'config': self.config,
            'vulnerability_types': self.config['vulnerability_types']
        }
        
        model_dir = Path(self.config['paths'].get('model_dir', './cabis_project/models'))
        final_path = model_dir / 'cabis_final.pt'
        torch.save(final_checkpoint, final_path)
        logger.info(f"Saved final model: {final_path}")
    
    def _should_stop_early(self, val_metrics):
        """Check if training should stop early"""
        # Simple patience-based early stopping
        patience = self.config.get('training', {}).get('early_stopping_patience', 150)
        
        if not hasattr(self, 'patience_counter'):
            self.patience_counter = 0
            self.best_val_loss = float('inf')
        
        if val_metrics['total_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['total_loss']
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= patience


def main():
    parser = argparse.ArgumentParser(description='Train CABIS on full SolidiFI dataset')
    parser.add_argument('--config', type=str, default='cabis_project/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='cabis_project/data/solidifi',
                       help='Path to SolidiFI dataset')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with small dataset')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.info("="*80)
    logger.info("CABIS Training on Full SolidiFI Dataset")
    logger.info("="*80)
    
    # Prepare dataset
    logger.info(f"\n📊 Loading SolidiFI dataset from: {args.data_dir}")
    dataset, patterns = prepare_solidifi_data(args.data_dir)
    
    if dataset is None or len(dataset) == 0:
        logger.error("Failed to load dataset!")
        return
    
    logger.info(f"✅ Loaded {len(dataset)} contract pairs")
    
    # Show vulnerability distribution
    vuln_counts = defaultdict(int)
    for i in range(len(dataset)):
        vuln_type = dataset[i].get('vulnerability_type', 'unknown')
        vuln_counts[vuln_type] += 1
    
    logger.info("\nVulnerability distribution:")
    for vuln_type, count in sorted(vuln_counts.items()):
        logger.info(f"  {vuln_type}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Split dataset
    if args.debug:
        # Use small subset for debugging
        dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
        logger.info("🐛 Debug mode: Using subset of 100 samples")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Val: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")
    
    # Initialize trainer
    logger.info(f"\n🚀 Initializing CABIS trainer with config: {args.config}")
    trainer = EnhancedCABISTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\n📂 Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        
        trainer.encoder.load_state_dict(checkpoint['encoder_state'])
        trainer.synthesizer.load_state_dict(checkpoint['synthesizer_state'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator_state'])
        trainer.analyzer.load_state_dict(checkpoint['analyzer_state'])
        
        if 'encoder_optimizer' in checkpoint:
            trainer.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            trainer.synthesizer_optimizer.load_state_dict(checkpoint['synthesizer_optimizer'])
            trainer.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            trainer.analyzer_optimizer.load_state_dict(checkpoint['analyzer_optimizer'])
        
        if 'schedulers' in checkpoint:
            for name, scheduler in trainer.schedulers.items():
                if name in checkpoint['schedulers']:
                    scheduler.load_state_dict(checkpoint['schedulers'][name])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"✅ Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Start training
    logger.info(f"\n🏋️ Starting training from epoch {start_epoch+1}")
    trainer.train(train_dataset, val_dataset, num_epochs=args.epochs)
    
    logger.info("\n✅ Training completed!")
    logger.info(f"Best model saved to: {trainer.config['paths'].get('model_dir', './cabis_project/models')}/cabis_best.pt")
    
    # Final evaluation on test set
    logger.info("\n📈 Final evaluation on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer.config['evaluation']['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    test_metrics = trainer._validate_epoch(test_loader)
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    if test_metrics.get('per_class'):
        logger.info("Test per-class F1 scores:")
        for vuln_type, score in test_metrics.get('per_class', {}).items():
            logger.info(f"  {vuln_type}: {score:.4f}")


if __name__ == "__main__":
    main()