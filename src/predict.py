# src/predict.py

import joblib
import pandas as pd


def load_model(model_path="models/best_model.pkl"):
    """
    Loads the saved ML pipeline.
    """
    model = joblib.load(model_path)
    return model


def make_predictions(model, csv_path):
    """
    Makes predictions on new/unseen data.
    """

    # Load new data
    data = pd.read_csv(csv_path)

    # Make predictions
    predictions = model.predict(data)

    return predictions
