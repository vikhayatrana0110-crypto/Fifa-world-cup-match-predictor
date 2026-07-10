# ⚽ World Cup AI Predictor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A machine learning analytics platform that predicts international association football match outcomes (Win/Draw/Loss probabilities and expected goals) using historical match results, rolling team statistics, and ELO ratings. Explains model decisions using SHAP value feature contributions.

---

## 🌟 Key Features

- **Match Prediction**: Generates home win, draw, and away win probabilities alongside Expected Goals (xG) estimates for both teams.
- **Explainability**: Integrated SHAP (SHapley Additive exPlanations) values to detail exactly why the model made a prediction (positive/negative contributors).
- **ELO Rating Engine**: Calculates dynamic team Elo ratings updated match-by-match chronologically.
- **Rolling Form & Stats**: Engineers team-level metrics (rolling attack/defense rating, weighted form score, average goals scored/conceded).
- **Interactive UI**: A dark-themed Streamlit interface featuring glassmorphic designs, Gauge indicators, radar charts, and interactive Plotly visualizations.
- **Robust Schema**: A structured 3-layer database architecture (Raw -> Feature -> ML) managed via SQLAlchemy.

---

## 🛠️ Tech Stack

- **Data Engineering**: `pandas`, `numpy`, `SQLAlchemy`, `SQLite`
- **Machine Learning**: `scikit-learn`, `xgboost`, `shap`, `joblib`
- **Frontend / Visualizations**: `streamlit`, `plotly`

---

## 📁 Repository Structure

```
world-cup-ai-predictor/
├── .streamlit/
│   └── config.toml          # Dark theme configuration
├── data/
│   └── raw/                 # Raw datasets (results.csv)
├── sql/
│   └── schema.sql           # Database schema SQL statements
├── src/
│   ├── app/
│   │   └── streamlit_app.py # Streamlit UI implementation
│   ├── database/
│   │   ├── models.py        # SQLAlchemy database models
│   │   └── db_manager.py    # CRUD and SQL database operations
│   ├── features/
│   │   ├── elo_calculator.py # Elo calculation logic
│   │   ├── team_features.py  # Team-level rolling features
│   │   └── match_features.py # Match-level difference features
│   └── models/
│       ├── trainer.py       # Model training pipeline
│       ├── predictor.py     # Inference interface
│       └── explainer.py     # SHAP explainer wrapper
├── tests/
│   ├── test_elo.py          # Unit tests for Elo calculator
│   ├── test_features.py     # Unit tests for feature engine
│   └── test_predictor.py    # Unit tests for prediction and SHAP
├── requirements.txt         # Package dependencies
└── README.md                # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment
First, ensure you have Python 3.8+ installed. Navigate to the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Database & Pipeline Ingestion
You can ingest historical results, calculate Elos, build feature tables, and train the XGBoost models with a single pipeline trigger directly from the Streamlit UI, or run it programmatically:
```bash
python -m src.ingestion.download_data
```

### 3. Launch Streamlit Application
Start the interactive UI:
```bash
streamlit run src/app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Running Tests
Verify database, features, and model prediction logic using `unittest`:
```bash
python -m unittest discover tests
```

---

## 📝 License
This project is licensed under the MIT License.
