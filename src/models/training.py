import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, List
import logging
from tqdm import tqdm

from ..utils.utils import save_checkpoint

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for motion sequence classification model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        label_mapping: Dict[str, int]
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            config: Configuration dictionary
            device: Device to train on
            label_mapping: Dictionary mapping labels to indices
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.label_mapping = label_mapping
        
        self.clip_value = config['training']['clip_value']
        self.patience = config['training']['patience']
        self.num_epochs = config['training']['num_epochs']
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for sequences, labels in progress_bar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_value
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc='Validation'):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            logger.info(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.training_history['train_losses'].append(train_loss)
            self.training_history['train_accs'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_accs'].append(val_acc)
            
            logger.info(
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n'
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Save checkpoint and check for early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch+1,
                loss=val_loss,
                accuracy=val_acc,
                config=self.config,
                label_mapping=self.label_mapping,
                is_best=is_best
            )
            
            if self.patience_counter >= self.patience:
                logger.info(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
        
        return self.training_history
    
    def predict(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on new sequences.
        
        Args:
            sequences: Input sequences tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            sequences = sequences.to(self.device)
            outputs = self.model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities

def get_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Create optimizer based on config.
    
    Args:
        model: The PyTorch model
        config: Configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

def get_criterion(config: Dict[str, Any]) -> nn.Module:
    """
    Create loss function based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch loss function
    """
    return nn.CrossEntropyLoss(
        label_smoothing=config['training']['label_smoothing']
    )