import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import pandas as pd
from models import LSTMModel
from preprocess import load_data, prepare_sequences

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# Load and preprocess
df = load_data(config["data"]["path"])
X, y, scaler = prepare_sequences(df, config["data"]["sequence_length"])

split_idx = int(len(X) * config["data"]["train_split"])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32)),
                          batch_size=config["training"]["batch_size"], shuffle=True)

# Model
model = LSTMModel(config["model"]["input_size"],
                  config["model"]["hidden_size"],
                  config["model"]["num_layers"],
                  config["model"]["dropout"]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Training loop
log = []
for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    log.append({"epoch": epoch+1, "loss": avg_loss})
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save logs and model
pd.DataFrame(log).to_csv(config["logging"]["log_file"], index=False)
torch.save(model.state_dict(), "results/lstm_model.pth")
