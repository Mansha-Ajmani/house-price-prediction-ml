# House Price Prediction Using Machine Learning

## Overview
This project develops a machine learning-based regression model to predict house prices using property-related features such as location, size, number of rooms, property age, and amenities.

The workflow includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Model training and optimization
- Performance evaluation
- Feature importance analysis using SHAP

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP

## Machine Learning Model
- Gradient Boosting Regressor
- Hyperparameter tuning using RandomizedSearchCV

## Evaluation Metrics
- R² Score
- RMSE (Root Mean Squared Error)

## Project Structure

```bash
house-price-prediction-ml/
│
├── data/
├── models/
├── outputs/
├── house_price_prediction.py
├── requirements.txt
└── README.md
```

## Features
- House price prediction
- Feature scaling and encoding
- Hyperparameter optimization
- SHAP-based feature importance analysis
- Model persistence using Joblib

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python house_price_prediction.py
```

## Applications
- Real estate analytics
- Investment analysis
- Property valuation
- Data-driven decision making

## Author
Mansha Ajmani
