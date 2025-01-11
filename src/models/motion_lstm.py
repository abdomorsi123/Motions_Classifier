import torch
import torch.nn as nn
from typing import Tuple, Optional

class MotionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super(MotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Additional processing layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        
        # Final classification layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with appropriate initialization schemes"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.orthogonal_(param)
                elif 'attention' in name:
                    # Attention weights
                    if param.dim() > 1:
                        nn.init.kaiming_normal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bn' in name:
                    # BatchNorm weights
                    nn.init.uniform_(param, 0.0, 1.0)
                elif 'fc' in name:
                    # Linear layers
                    if param.dim() > 1:
                        nn.init.kaiming_normal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        # Apply batch norm to input features
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Multi-head attention
        attention_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attention_out = attention_out.transpose(0, 1)
        
        # Global average pooling
        pooled = torch.mean(attention_out, dim=1)
        
        # Final processing
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        attention_out, attention_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        return attention_out.transpose(0, 1), attention_weights
