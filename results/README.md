# Results

This folder contains outputs from training and evaluation of the ML-Finance project.

## Files
- **training_log.csv**  
  Logs training progress (epoch, training loss, validation loss, MAE). Useful for debugging and reproducing results

- **metrics.png**  
  Visualization of model performance across epochs (training vs validation loss). Helps show convergence and generalization

- **predictions.csv**  
  Model forecasts vs. actual NASDAQ closing prices for December 2024  
  Columns: `Date`, `Actual_Close`, `Predicted_Close`

- **predictions_chart.png**  
  Visualization of `predictions.csv`, showing Actual vs Predicted NASDAQ closes over time

## Summary
The model converged smoothly (validation loss ~0.0125 by epoch 10) and achieved a mean absolute error (MAE) under ~80 points on the NASDAQ index, which is strong for a limited dataset.  
The predictions closely track the real market trend, showing the model captured price dynamics reasonably well.

## Notes
- Results are based on the dataset: `data/nasdaq_dec2024.csv`
- For future experiments, create subfolders under `results/` (e.g., `results/experiment_1/`) to keep logs, charts, and predictions organized
