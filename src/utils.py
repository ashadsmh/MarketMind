import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_log(log, path):
    pd.DataFrame(log).to_csv(path, index=False)
