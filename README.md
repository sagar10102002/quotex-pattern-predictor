
# Quotex Pattern Predictor

A machine learning project that predicts the next 10 UP/DOWN patterns in Quotex using LSTM.

## Features
- Simulated Quotex-like dataset
- LSTM-based sequence model
- Streamlit dashboard for predictions
- Win/Loss simulator and balance tracker

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train model:
```
python model/train_model.py
```

3. Run app:
```
streamlit run app/dashboard.py
```

## File Structure
- `data/simulated_quotex_data.csv` — Simulated trading data
- `model/train_model.py` — LSTM model training
- `app/dashboard.py` — Streamlit frontend
- `utils/sequence_generator.py` — Preprocessing utility
