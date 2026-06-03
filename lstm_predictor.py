import os
import logging
import numpy as np

# PyTorch import safety guard
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object

logger = logging.getLogger(__name__)
MODEL_WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_orbit_model.pt")

if nn is not object:
    class SatelliteLSTMPredictor(nn.Module):
        """
        LSTM network designed to forecast orbital coordinates 7 days ahead
        based on a sequence of historical position states.
        """
        def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
            super(SatelliteLSTMPredictor, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_dim)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out
else:
    class SatelliteLSTMPredictor:
        pass

def predict_7d_position(state_sequence: list) -> np.ndarray:
    """
    Predicts the satellite's position at T+7 days based on the last 5 days sequence.
    If no pre-trained weights file exists or PyTorch is not available, uses an analytical
    orbital Keplerian model fallback.
    """
    if torch is None or not os.path.exists(MODEL_WEIGHTS_FILE):
        # Fallback to Keplerian propagation approximation (simulated forecast)
        logger.info("LSTM model weights or PyTorch missing. Using Keplerian orbital drift projection for 7-day forecast.")
        last_state = np.array(state_sequence[-1]) # Last [x, y, z] coordinate
        
        # Simulate standard orbit circular drift over 7 days (604800 seconds)
        # Shift coordinate position based on an assumed LEO orbital speed
        t = 7.0 * 86400.0
        n = 0.0011 # mean motion
        theta = n * t
        
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        predicted_pos = np.dot(rotation_matrix, last_state)
        return predicted_pos
        
    try:
        # Load model and run inference
        model = SatelliteLSTMPredictor()
        model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE, map_location=torch.device('cpu')))
        model.eval()
        
        # Prepare inputs
        inputs = torch.tensor([state_sequence], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs)
            
        return outputs.numpy()[0]
    except Exception as e:
        logger.error(f"Failed to execute LSTM 7-day prediction: {e}. Falling back.")
        return np.array(state_sequence[-1])

def train_lstm_model(training_data_x, training_data_y, epochs=10):
    """
    Trains the LSTM model on historical coordinate matrices and saves weights.
    """
    if torch is None:
        logger.error("PyTorch not installed. Cannot train LSTM model.")
        return
        
    try:
        model = SatelliteLSTMPredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensors
        inputs = torch.tensor(training_data_x, dtype=torch.float32)
        targets = torch.tensor(training_data_y, dtype=torch.float32)
        
        print("Starting training of LSTM 7-day Orbit Predictor...")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
        # Save weights
        torch.save(model.state_dict(), MODEL_WEIGHTS_FILE)
        print(f"Model saved successfully to {MODEL_WEIGHTS_FILE}")
    except Exception as e:
        logger.error(f"Error during LSTM training: {e}")
