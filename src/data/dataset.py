import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class MotionSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[str],
        max_len: Optional[int] = None
    ):
        """
        Initialize the dataset
        
        Args:
            sequences: List of numpy arrays, each array is a sequence
            labels: List of strings (actions)
            max_len: Maximum sequence length (if None, use length of longest sequence)
        """
        self.max_len = max_len or max(len(seq) for seq in sequences)
        
        # Process sequences
        self.sequences = self._process_sequences(sequences)
        
        # Convert string labels to numeric
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.labels = [self.label_to_idx[label] for label in labels]
    
    def _process_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """Process sequences to have fixed length"""
        from .preprocessing import DataPreprocessor
        
        processed_sequences = []
        for seq in sequences:
            # Normalize the sequence
            normalized_seq = DataPreprocessor.normalize_sequence(seq)
            
            # Pad/truncate to fixed length
            processed_seq = DataPreprocessor.pad_sequence(normalized_seq, self.max_len)
            processed_sequences.append(processed_seq)
        
        return processed_sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sequence, label
    
    def get_num_classes(self) -> int:
        return len(self.label_to_idx)
    
    def get_label_mapping(self) -> Dict[str, int]:
        return self.label_to_idx

def load_sequences(config: Dict[str, Any], dataset: str = 'train') -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all sequences from the specified directory
    
    Args:
        config: Configuration dictionary
        dataset: Either 'train' or 'test'
        
    Returns:
        Tuple of (sequences, labels)
    """
    sequences = []
    labels = []
    sequence_lengths = []
    
    data_dir = Path(config['data']['base_path']) / config['data'][f'{dataset}_dir']
    csv_files = list(data_dir.glob("*.csv"))
    
    logger.info(f"Found {len(csv_files)} sequences in {dataset} set")
    
    for file_path in csv_files:
        # Extract action from filename
        action = file_path.stem.split('_')[1]
        
        # Load sequence
        sequence = pd.read_csv(file_path, header=None).values
        sequences.append(sequence)
        labels.append(action)
        sequence_lengths.append(len(sequence))
    
    # Log sequence statistics
    logger.info("\nSequence length statistics:")
    logger.info(f"Min length: {min(sequence_lengths)}")
    logger.info(f"Max length: {max(sequence_lengths)}")
    logger.info(f"Mean length: {np.mean(sequence_lengths):.2f}")
    logger.info(f"Median length: {np.median(sequence_lengths):.2f}")
    
    return sequences, labels

def create_data_loaders(
    config: Dict[str, Any],
    sequences: List[np.ndarray],
    labels: List[str]
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Split data into train/val and create DataLoaders
    
    Args:
        config: Configuration dictionary
        sequences: List of sequences
        labels: List of labels
        
    Returns:
        Tuple of (train_loader, val_loader, label_mapping)
    """
    # Split into train and validation
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels,
        test_size=config['dataset']['val_split'],
        shuffle=config['dataset']['shuffle'],
        stratify=labels,
        random_state=42
    )
    
    # Create datasets
    train_dataset = MotionSequenceDataset(
        train_sequences,
        train_labels,
        max_len=config['data']['max_sequence_length']
    )
    val_dataset = MotionSequenceDataset(
        val_sequences,
        val_labels,
        max_len=config['data']['max_sequence_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers']
    )
    
    return train_loader, val_loader, train_dataset.get_label_mapping()