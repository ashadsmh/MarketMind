# Data

This folder contains raw and processed data used for the ML-Finance project.

## Files
- **nasdaq_dec2024.csv**  
  Daily NASDAQ index values (Open, High, Low, Close, Adjusted Close, Volume) for December 2024.  
  Source: Yahoo Finance (manually downloaded).  

## Structure
- `raw/` or base folder → contains original CSV files as downloaded
- `processed/` → contains cleaned datasets or feature-engineered versions created during preprocessing

## Notes
- Keep raw data immutable (do not overwrite original CSVs)
- Place any new market data here (e.g., other months or tickers)
- Preprocessing scripts will read from `data/` and write to `data/processed/`
