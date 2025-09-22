# Results

This folder contains experiment outputs and evaluation artifacts for the ML-Finance project.

## Files
- **training_log.csv**  
  Logs training progress (epoch, training loss, validation loss, accuracy/MAE). Useful for debugging and reproducing results.

- **metrics.png**  
  Visualization of model performance across epochs (e.g., training vs validation loss). Helps show convergence and generalization.

- **predictions.csv**  
  Model forecasts vs. actual stock prices for December 2024.  
  Columns: `Date`, `Actual_Close`, `Predicted_Close`.

## Notes
- Results here are based on the dataset `data/nasdaq_dec2024.csv`.
- For future experiments, create subfolders:  
