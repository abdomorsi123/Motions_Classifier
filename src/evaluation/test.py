import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from ..data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        label_mapping: Dict[str, int],
        device: torch.device
    ):
        """
        Initialize the model tester
        
        Args:
            model: Trained PyTorch model
            config: Configuration dictionary
            label_mapping: Dictionary mapping label names to indices
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.label_mapping = label_mapping
        self.device = device
        self.idx_to_label = {v: k for k, v in label_mapping.items()}
        
        # Initialize preprocessor but don't save files
        self.preprocessor = DataPreprocessor(config)
    
    def preprocess_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Preprocess a single sequence without saving
        
        Args:
            sequence: Raw sequence data
            
        Returns:
            Preprocessed sequence
        """
        # Apply preprocessing steps without saving
        
        # Remove rows with only 75 features
        sequence = sequence[~(np.sum(np.isnan(sequence[:, -4:]), axis=1) == 4)]
        
        if self.config['data']['remove_legs']:
            # Remove leg-related columns
            legs_columns = list(range(30, 36)) + list(range(39, 45)) + list(range(57, 75))
            sequence = np.delete(sequence, legs_columns, axis=1)
        
        # Normalize
        sequence = DataPreprocessor.normalize_sequence(sequence)
        
        # Pad/truncate
        sequence = DataPreprocessor.pad_sequence(
            sequence,
            self.config['data']['max_sequence_length']
        )
        
        return sequence
    
    def load_and_preprocess_test_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load and preprocess all test sequences
        
        Returns:
            Tuple of (preprocessed_sequences, labels)
        """
        sequences = []
        labels = []
        test_dir = Path(self.config['data']['base_path']) / self.config['data']['test_dir']
        
        logger.info("Loading test sequences...")
        for file_path in test_dir.glob("*.csv"):
            # Extract label from filename
            action = file_path.stem.split('_')[1]
            
            # Only process if action is in label mapping
            if action in self.label_mapping:
                # Load and preprocess sequence
                sequence = pd.read_csv(file_path, names=list(range(79))).values
                processed_sequence = self.preprocess_sequence(sequence)
                
                sequences.append(processed_sequence)
                labels.append(action)
            else:
                logger.warning(f"Skipping unknown action {action} in {file_path}")
        
        logger.info(f"Loaded {len(sequences)} test sequences")
        return sequences, labels
    
    def predict_sequence(self, sequence: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Make prediction on a single sequence
        
        Args:
            sequence: Preprocessed sequence data
            
        Returns:
            Tuple of (predicted_label, confidence_scores)
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare input
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Get model prediction
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            
            # Get confidence scores for all classes
            confidence_scores = {
                self.idx_to_label[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
            
            predicted_label = self.idx_to_label[predicted_idx]
            
        return predicted_label, confidence_scores
    
    def evaluate_test_set(self) -> Dict[str, Any]:
        """
        Evaluate model on entire test set
        
        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        # Load and preprocess test data
        sequences, true_labels = self.load_and_preprocess_test_data()
        
        # Make predictions
        predictions = []
        confidence_scores = []
        
        logger.info("Making predictions on test set...")
        for sequence in sequences:
            pred_label, conf_scores = self.predict_sequence(sequence)
            predictions.append(pred_label)
            confidence_scores.append(conf_scores)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(
            [self.label_mapping[label] for label in true_labels],
            [self.label_mapping[label] for label in predictions]
        )
        
        # Generate classification report
        class_report = classification_report(
            true_labels,
            predictions,
            output_dict=True
        )
        
        return {
            'true_labels': true_labels,
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray) -> None:
        """
        Plot confusion matrix
        
        Args:
            confusion_mat: Confusion matrix array
        """
        plt.figure(figsize=(12, 10))
        labels = list(self.label_mapping.keys())
        
        sns.heatmap(
            confusion_mat,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()