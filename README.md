# Machine learning project 


Developed as a technical assessment covering financial data engineering, time-series validation, machine learning, and API development.

Overview

The model predicts whether Gold's next trading-day closing price will be:

UP — next close is higher than the current close
DOWN — next close is equal to or lower than the current close

The project pipeline:

Data → Feature Engineering → Experiments → Training → Prediction → FastAPI

Data

Historical Gold Futures OHLC data is downloaded from Yahoo Finance using yfinance.

The project uses GC=F as a proxy for XAUUSD spot prices, since Yahoo Finance does not provide direct XAUUSD historical data through the same interface.

The dataset contains:

Open
High
Low
Close
Adjusted Close
Volume

Data is stored in:

data/raw/xauusd.csv
Target

The target represents the direction of the next trading day's closing price:

1 = UP
0 = DOWN

It is calculated as:

df["target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

The final row is removed because there is no future closing price available to calculate its target.

Features

The model uses 10 features:

Feature	Description
return_1d	One-day price return
return_5d	Five-day price return
return_10d	Ten-day price return
price_vs_sma_10	Price relative to 10-day SMA
price_vs_sma_30	Price relative to 30-day SMA
sma_10_vs_sma_30	Short vs. medium-term trend
volatility_10	10-day rolling volatility
volatility_20	20-day rolling volatility
volume_change	Daily volume change
price_position_20	Price position within the 20-day range

All features use information available at or before the prediction date, avoiding look-ahead leakage.

Time-Based Validation

Because this is financial time-series data, the dataset is not randomly shuffled.

The first 80% is used for training and the final 20% for testing.

	Period	Samples
Training	2015-02-13 → 2024-04-08	2,268
Testing	2024-04-09 → 2026-07-21	568

This ensures the model is evaluated on a future period that occurs strictly after the training data.

Model Comparison

Three classifiers were evaluated:

Logistic Regression
Random Forest
Gradient Boosting

A majority-class baseline was also included.

Results
Model	Accuracy	ROC-AUC	Precision	Recall	F1
Logistic Regression	0.5299	0.5232	0.5696	0.6781	0.6191
Random Forest	0.4930	0.5088	0.5611	0.4594	0.5052
Gradient Boosting	0.5053	0.5013	0.5582	0.5844	0.5710
Baseline	0.5634	—	—	—	—
Interpretation

The experiments show that the tested models did not outperform the baseline in accuracy.

Logistic Regression achieved the best overall machine learning results, but the current feature set does not demonstrate a strong predictive edge.

This is an intentional part of the evaluation: the project prioritizes honest time-based validation and transparent results rather than presenting an artificially strong model.

Prediction Horizon Experiment

The project also tested prediction horizons of:

1 trading day
3 trading days
5 trading days

The experiments did not show a consistent improvement over the corresponding baselines.

This suggests that the current feature set does not provide a strong directional signal across the tested horizons.

Project Structure
gold-direction-predictor/
│
├── app/
│   ├── api.py
│   ├── download_data.py
│   ├── features.py
│   ├── experiment.py
│   ├── experiment_horizon.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   └── raw/
│       └── xauusd.csv
│
├── models/
│   └── gold_direction_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
Installation

Create and activate a virtual environment:

python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Usage
1. Download market data
python app/download_data.py
2. Generate features
python app/features.py
3. Run model experiments
python app/experiment.py

To test different prediction horizons:

python app/experiment_horizon.py
4. Train the final model
python app/train.py

The trained model is saved to:

models/gold_direction_model.pkl
5. Generate a prediction
python app/predict.py

Example output:

Prediction: DOWN
Probability of UP: 49.09%
Probability of DOWN: 50.91%
FastAPI

Start the API with:

uvicorn app.api:app --reload

The API runs at:

http://127.0.0.1:8000
Health Check
GET /health

Example:

{
  "status": "ok",
  "model_loaded": true
}
Prediction
GET /predict

Example:

{
  "date": "2026-07-21",
  "close": 4088.9,
  "prediction": "DOWN",
  "probability_up": 0.4909,
  "probability_down": 0.5091
}

Interactive API documentation:

http://127.0.0.1:8000/docs
Limitations

This project is a machine learning experiment and not financial advice or a trading strategy.

Current limitations include:

Gold Futures are used as a proxy for XAUUSD spot.
No macroeconomic or news data is included.
No transaction costs or slippage are modeled.
The current models do not outperform the baseline.
Short-term Gold price direction remains highly noisy.

The results should therefore be interpreted as an evaluation of the data and ML pipeline, not as evidence of a profitable trading system.

Future Improvements

Potential improvements include:

Adding macroeconomic data and DXY
Testing additional technical indicators
Time-series cross-validation
Hyperparameter optimization
Probability calibration
Backtesting with transaction costs and slippage
Market regime detection
Exploring alternative targets and prediction horizons
Conclusion

This project demonstrates an end-to-end machine learning pipeline for financial time-series data, including:

Data collection → Feature engineering → Time-based validation → Model experimentation → Training → Prediction → API deployment

The experiments also highlight an important principle in financial ML: honest evaluation matters more than achieving an artificially high metric.