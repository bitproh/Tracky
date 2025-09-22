# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
INPUT_SEQUENCE_LENGTH = 20  # Use 20 past points to predict
OUTPUT_SEQUENCE_LENGTH = 50 # Predict 50 future points
TRAINING_EPOCHS = 100

# --- DATA PREPARATION ---
print("Loading and preparing trajectory data...")
df = pd.read_csv('trajectories.csv')
scaler = MinMaxScaler()
df[['x', 'y']] = scaler.fit_transform(df[['x', 'y']])

sequences =
for track_id in df['track_id'].unique():
    track_data = df[df['track_id'] == track_id][['x', 'y']].values
    if len(track_data) > INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH:
        for i in range(len(track_data) - INPUT_SEQUENCE_LENGTH - OUTPUT_SEQUENCE_LENGTH):
            input_seq = track_data
            output_seq = track_data
            sequences.append((input_seq, output_seq))

# Convert to PyTorch tensors
input_tensors = torch.FloatTensor(np.array([s for s in sequences]))
output_tensors = torch.FloatTensor(np.array([s[2] for s in sequences]))

# --- LSTM MODEL DEFINITION ---
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, future=OUTPUT_SEQUENCE_LENGTH):
        outputs =
        # Get the hidden state from the input sequence
        _, (h_n, c_n) = self.lstm(x)
        
        # Use the last point of the input sequence as the first input for the decoder
        decoder_input = x[:, -1, :]
        
        # Predict future points
        for _ in range(future):
            output, (h_n, c_n) = self.lstm(decoder_input.unsqueeze(1), (h_n, c_n))
            output = self.linear(output.squeeze(1))
            outputs.append(output)
            decoder_input = output # Use the prediction as the next input
            
        return torch.stack(outputs, 1)

# --- TRAINING LOOP ---
print("Training LSTM model...")
model = TrajectoryLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(TRAINING_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_tensors)
    loss = criterion(outputs, output_tensors)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch, Loss: {loss.item():.6f}')

# --- SAVE THE MODEL ---
torch.save(model.state_dict(), 'lstm_predictor.pth')
print("Model training complete. Saved to lstm_predictor.pth")
