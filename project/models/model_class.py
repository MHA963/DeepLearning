# models.py
import torch
import torch.nn as nn

import torch
import torch.nn as nn

# --- 1. Basic CNN-LSTM Model (Renamed from CNN_LSTM_Model_7) ---
# Input: (B, L, 7) -> CNN (32, 64 filters) -> LSTM (64 hidden)
class Basic_CNN_LSTM(nn.Module):
    """
    A basic CNN-LSTM model expecting 7 input features (e.g., OHLCV + 2 indicators).
    Uses 32 and 64 filters and a 64-unit LSTM layer.
    """
    def __init__(self, input_dim=7, lookback=48):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1) 
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
        # LSTM layer (input size matches Conv2 output channels)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        
        # Dense head
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Data Permute: (B, L, C) -> (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # CNN Blocks
        x = torch.relu(self.conv1(x))
        x = self.pool(self.dropout1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(self.dropout2(x))
        
        # Permute Back: (B, C, L_reduced) -> (B, L_reduced, C) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM and Dense Head
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last timestep output
        out = self.dropout3(out)
        out = torch.relu(self.fc1(out))
        
        # Final Sigmoid for Probability Output
        out = torch.sigmoid(self.fc2(out)) 
        return out

# --- 2. Optimised CNN-LSTM Model (Renamed from CNN_LSTM_Model_v03) ---
# Added BatchNorm for stability and used the standard 32/64 filter sizes.
class Optimised_CNN_LSTM(nn.Module):
    """
    An optimised CNN-LSTM model incorporating BatchNorm layers for improved stability 
    and training speed. Expects 7 input features.
    """
    def __init__(self, input_dim=7, lookback=48):
        super().__init__()
        
        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d( input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2) 
        self.drop = nn.Dropout(0.3) # Dropout applied after pooling
        
        # LSTM layer (input size matches Conv2 output channels)
        self.lstm = nn.LSTM(64, 64, batch_first=True)

        # Dense head
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, features, seq)

        # CNN Block 1: Conv -> BN -> ReLU -> Drop -> Pool
        # ✅ FIX: Using torch.relu() instead of instantiating nn.ReLU()
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.drop(x))

        # CNN Block 2: Conv -> BN -> ReLU -> Drop -> Pool
        # ✅ FIX: Using torch.relu() instead of instantiating nn.ReLU()
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(self.drop(x))

        x = x.permute(0, 2, 1) # back to (batch, seq, features)

        x, _ = self.lstm(x)
        x = x[:, -1, :]       # last timestep

        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        
        # Returns Sigmoid probability
        return torch.sigmoid(x)

# --- 3. Optimised Large CNN-LSTM Model (Renamed from Optimized_CNN_LSTM_Model) ---
# Increased filter counts (64, 128) and LSTM hidden size (128).
class Optimised_Large_CNN_LSTM(nn.Module):
    """
    A larger, optimised CNN-LSTM model with increased capacity: 
    64/128 filters and 128-unit LSTM. Expects 7 input features.
    """
    def __init__(self, input_dim=7, lookback=48):
        super().__init__()
        
        # Increased filter counts
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1) 
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Adjusted Dropout rates
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        
        # Increased LSTM hidden size (input size matches Conv2 output channels: 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True) 
        
        self.dropout3 = nn.Dropout(0.3)
        
        # Adjusted Dense layer input to match new LSTM size (128)
        self.fc1 = nn.Linear(128, 64) 
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Initial permute: (B, L, C) -> (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # CNN Block 1 (64 filters)
        x = torch.relu(self.conv1(x))
        x = self.pool(self.dropout1(x))
        
        # CNN Block 2 (128 filters)
        x = torch.relu(self.conv2(x))
        x = self.pool(self.dropout2(x))
        
        # Permute back: (B, C, L_reduced) -> (B, L_reduced, C_new=128) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layer (128 hidden units)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        # Dense Classification Head
        out = self.dropout3(out)
        out = torch.relu(self.fc1(out))
        
        # Final Sigmoid for Probability Output
        out = torch.sigmoid(self.fc2(out))
        return out
    
class Basic_CNN_LSTM_Tuned(nn.Module):
    """
    A Basic CNN-LSTM model where the LSTM hidden size can be easily adjusted.
    """
    def __init__(self, input_dim=7, lookback=48, lstm_hidden_size=64):
        super().__init__()
        # CNN layers - Fixed (32, 64 filters)
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1) 
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
        # 🎯 UPDATED: LSTM hidden size is now a variable
        self.lstm = nn.LSTM(64, lstm_hidden_size, batch_first=True)
        
        # Dense head (input size must match LSTM hidden size)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc2 = nn.Linear(lstm_hidden_size // 2, 1)
        
    def forward(self, x):
        # ... (rest of the forward pass is the same as Basic_CNN_LSTM)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(self.dropout1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(self.dropout2(x))
        x = x.permute(0, 2, 1)
        
        # LSTM and Dense Head
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last timestep output
        out = self.dropout3(out)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out)) 
        return out

class Base_line_LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return torch.sigmoid(out)
    
class CNN_LSTM_Model_5(nn.Module):
    def __init__(self, input_dim=5, lookback=48):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1) 
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32,1)
        
    def forward(self, x):
        # Data Permute: (B, L, C) -> (B, C, L)
        x = x.permute(0,2,1)
        
        # CNN Blocks
        x = torch.relu(self.conv1(x))
        x = self.pool(self.dropout1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(self.dropout2(x))
        
        # Permute Back: (B, C, L) -> (B, L, C)
        x = x.permute(0,2,1)
        
        # LSTM and Dense Head
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last timestep output
        out = self.dropout3(out)
        out = torch.relu(self.fc1(out))
        # Final Sigmoid for Probability Output
        out = torch.sigmoid(self.fc2(out)) 
        return out