import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.base_path = Path(config['data']['base_path'])
        self.remove_legs = config['data']['remove_legs']
        
        # Define keypoint indices (each keypoint has x, y, c)
        self.keypoint_indices = {
            'nose': 0,          # 0,1,2
            'neck': 3,          # 3,4,5
            'rshoulder': 6,     # 6,7,8
            'relbow': 9,        # 9,10,11
            'rwrist': 12,       # 12,13,14
            'lshoulder': 15,    # 15,16,17
            'lelbow': 18,       # 18,19,20
            'lwrist': 21,       # 21,22,23
            'mhip': 24,         # 24,25,26
            'rhip': 27,         # 27,28,29
            'lhip': 36,         # 36,37,38
            'reye': 45,         # 45,46,47
            'leye': 48,         # 48,49,50
            'rear': 51,         # 51,52,53
            'lear': 54,         # 54,55,56
        }
        
        if not self.remove_legs:
            self.keypoint_indices.update({
                'rknee': 30,        # 30,31,32
                'rankle': 33,       # 33,34,35
                'lknee': 39,        # 39,40,41
                'lankle': 42,       # 42,43,44
                'lbigtoe': 57,      # 57,58,59
                'lsmalltoe': 60,    # 60,61,62
                'lheel': 63,        # 63,64,65
                'rbigtoe': 66,      # 66,67,68
                'rsmalltoe': 69,    # 69,70,71
                'rheel': 72         # 72,73,74
            })
    
    def preprocess_file(self, file_path: Path) -> None:
        """
        Preprocess a single CSV file
        
        Args:
            file_path: Path to the CSV file
        """
        try:
            logger.info(f"Processing {file_path}")
            
            # Read all data
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    values = line.strip().split(',')
                    if len(values) == 79:  # Only keep rows with all features
                        data.append([float(v) for v in values])
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if self.remove_legs:
                # Remove leg-related columns
                legs_columns = list(range(30, 36)) + list(range(39, 45)) + list(range(57, 75))
                df = df.drop(columns=legs_columns)
            
            # Save preprocessed data
            df.to_csv(file_path, index=False, header=False)
            logger.info(f"Successfully preprocessed {file_path}")
            
        except Exception as e:
            logger.error(f"Error preprocessing file {file_path}: {str(e)}")
            raise
    
    def preprocess_all_data(self) -> None:
        """Preprocess all CSV files in both train and test directories"""
        for dataset_type in [self.config['data']['train_dir']]: # , self.config['data']['test_dir']
            data_dir = self.base_path / dataset_type
            csv_files = list(data_dir.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {data_dir}")
                continue
            
            logger.info(f"Found {len(csv_files)} files in {dataset_type} directory")
            for file_path in csv_files:
                self.preprocess_file(file_path)
    
    @staticmethod
    def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
        """
        Normalize a sequence by centering and scaling
        
        Args:
            sequence: Input sequence array
            
        Returns:
            Normalized sequence
        """
        # Center the sequence
        mean = np.mean(sequence, axis=0)
        centered = sequence - mean
        
        # Scale the sequence
        std = np.std(centered, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = centered / std
        
        return normalized
    
    @staticmethod
    def pad_sequence(sequence: np.ndarray, max_len: int) -> np.ndarray:
        """
        Pad or truncate sequence to specified length
        
        Args:
            sequence: Input sequence array
            max_len: Target length
            
        Returns:
            Padded/truncated sequence
        """
        if len(sequence) > max_len:
            return sequence[:max_len]
        elif len(sequence) < max_len:
            padding_length = max_len - len(sequence)
            return np.pad(sequence, ((0, padding_length), (0, 0)), mode='constant')
        return sequence