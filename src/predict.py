import torch
import yaml
import pandas as pd
from models import LSTMModel
from preprocess import prepare_sequences

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

def predict_new(dataframe):
    X, _, scaler = prepare_sequences(dataframe, config["data"]["sequence_length"])

    model = LSTMModel(config["model"]["input_size"],
                      config["model"]["hidden_size"],
                      config["model"]["num_layers"],
                      config["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load("results/lstm_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    return predictions, scaler
