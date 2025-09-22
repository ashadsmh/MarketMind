import torch
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from models import LSTMModel
from preprocess import load_data, prepare_sequences

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

df = load_data(config["data"]["path"])
X, y, scaler = prepare_sequences(df, config["data"]["sequence_length"])

split_idx = int(len(X) * config["data"]["train_split"])
X_test, y_test = X[split_idx:], y[split_idx:]

model = LSTMModel(config["model"]["input_size"],
                  config["model"]["hidden_size"],
                  config["model"]["num_layers"],
                  config["model"]["dropout"]).to(device)
model.load_state_dict(torch.load("results/lstm_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    predictions = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

# Save predictions
pd.DataFrame({"True": y_test, "Predicted": predictions.squeeze()}).to_csv("results/predictions.csv", index=False)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test, label="True")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.savefig("results/metrics.png")
plt.close()
