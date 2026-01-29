import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    """
    Bahdanau-style Attention Mechanism.
    Helps the model focus on critical past time steps (e.g. recent volatility vs old trend).
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_dim] (last hidden state)
        # encoder_outputs: [batch, seq_len, hidden_dim] (all hidden states)
        
        # Energy: score(hidden, encoder_outputs)
        timestep_check = encoder_outputs.size(1)
        # Repeat hidden state seq_len times
        hidden_repeated = hidden.unsqueeze(1).repeat(1, timestep_check, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn(encoder_outputs)) # Simplified attention
        
        # In a real Bahdanau, we'd combine hidden and encoder_outputs. 
        # Here we use a self-attention simplified variant for stability on small datasets.
        
        # Calculate attention weights
        # We project energy to 1 dimension: [batch, seq_len, hidden_dim] -> [batch, seq_len]
        energy = energy.matmul(self.v) 
        attn_weights = torch.softmax(energy, dim=1) # [batch, seq_len]
        
        # Context vector: sum(weights * encoder_outputs)
        # [batch, 1, seq_len] bmm [batch, seq_len, hidden_dim] -> [batch, 1, hidden_dim]
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        return context_vector, attn_weights

class LSTMQuantAgent(nn.Module):
    """
    Advanced LSTM Model for Financial Time Series.
    Features:
    - Multi-layer LSTM
    - Dropout for regularization (and Monte Carlo Uncertainty)
    - Attention Head for interpretability
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMQuantAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.attention = Attention(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        # Initial hidden/cell states are zero by default
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # lstm_out: [batch, seq_len, hidden_dim]
        # h_n: [num_layers, batch, hidden_dim]
        
        # Use simple last-hidden state approach OR attention
        # We use Attention
        final_hidden_state = h_n[-1] # User the last layer's hidden state
        
        context_vector, attn_weights = self.attention(final_hidden_state, lstm_out)
        
        # context_vector: [batch, 1, hidden_dim]
        context_vector = context_vector.squeeze(1)
        
        prediction = self.fc(context_vector)
        return prediction

    def predict_with_uncertainty(self, x, n_samples=10):
        """
        Monte Carlo Dropout Inference.
        Runs the model multiple times with dropout enabled to estimate uncertainty.
        """
        self.train() # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(x).cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_dev = predictions.std(axis=0) # This is our "Uncertainty" / Inverse Confidence
        
        return mean_pred, std_dev
