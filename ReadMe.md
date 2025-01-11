# Motion Classification using Deep Learning

## Overview

This project implements a deep learning-based system for human motion classification using skeletal data. The model can classify different actions like boxing, playing drums, playing guitar, rowing, and playing violin with high accuracy (88.85% on the test set). It utilizes a bidirectional LSTM architecture with attention mechanism for sequence processing.

### Key Features

- Bidirectional LSTM with attention mechanism
- Real-time motion sequence processing
- Comprehensive data preprocessing pipeline
- Skeleton-based visualization system
- Detailed performance metrics and analysis
- Configurable training parameters via YAML

## Project Structure

```
project/
├── config/
│   └── config.yaml       # Configuration settings
│
├── src/
│   ├── data/             # Data processing modules
│   ├── visualization/    # Visualization tools
│   ├── models/           # Model architecture
│   └── utils/            # Utility functions
│
├── data/                 # folder contain the train and test data
│
├── frames/               # folder to save the output frames
│
├── videos/               # folder to save the output videos
│
├── checkpoints/          # folder to save the output models
│
├── results/
│   └── results.json      # the results from 5 individual runs
│
└── MotionClassifier.ipynb        # Main training notebook
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/motion-classification.git
cd motion-classification
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

The following packages are required (included in requirements.txt):

```
torch>=1.9.0
numpy>=1.19.2
pandas>=1.2.4
matplotlib>=3.3.4
seaborn>=0.11.1
opencv-python>=4.5.3
pyyaml>=5.4.1
scikit-learn>=0.24.2
tqdm>=4.61.0
```

## Usage

### Data Preparation

1. Place your motion sequence data in the following structure:

```
data/
├── train/
│   └── *.csv
└── test/
    └── *.csv
```

2. Each CSV file should contain skeletal coordinates with the format:
   - Columns: [x1, y1, c1, x2, y2, c2, ...]
   - Where (x,y) are coordinates and c is confidence score
   - One row per frame

### Running the Model

1. Configure the model:

   - Open `config/config.yaml`
   - Adjust parameters as needed (batch size, learning rate, etc.)

2. Training:
   - Open `MotionClassifier.ipynb`
   - Run all cells in order
   - The notebook is divided into sections:
     1. Setup and Configuration
     2. Data Processing
     3. Visualization
     4. Model Architecture
     5. Training
     6. Evaluation and Analysis

### Model Configuration

Key parameters in `config.yaml`:

```yaml
model:
  input_size: 49
  hidden_size: 256
  num_layers: 3
  dropout: 0.3
  num_attention_heads: 4

training:
  num_epochs: 100
  learning_rate: 0.0005
  batch_size: 16
```

## Results

The model achieves:

- Training Accuracy: 99.89%
- Validation Accuracy: 98.72%
- Test Accuracy: 88.85%

Per-class performance:

- Boxing: 95.16% F1-score
- Drums: 83.58% F1-score
- Guitar: 75.00% F1-score
- Rowing: 99.19% F1-score
- Violin: 90.60% F1-score

## Model Architecture

The system uses a MotionLSTM architecture with:

- Bidirectional LSTM layers
- Multi-head attention mechanism
- Batch normalization
- Residual connections
- Dropout regularization

### Data Flow

1. Input sequence → Batch Normalization
2. Normalized data → Bidirectional LSTM
3. LSTM output → Multi-head Attention
4. Attention output → Global Average Pooling
5. Pooled features → Fully Connected Layers
6. Output → Action Classification

## Visualization

The project includes tools for:

- Skeleton visualization
- Motion sequence rendering
- Performance metrics plotting
- Confusion matrix visualization

## Troubleshooting

### Common Issues

1. CUDA out of memory:

   - Reduce batch size in config.yaml
   - Reduce model size (hidden_size, num_layers)

2. Poor performance:

   - Check data preprocessing
   - Adjust learning rate
   - Increase model capacity
   - Implement data augmentation

3. Slow training:
   - Enable GPU acceleration
   - Increase num_workers in data loader
   - Optimize sequence length
