# src/model_selection.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


def get_models():
    """
    Returns a dictionary of ML models to evaluate.
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True)
    }

    return models


def evaluate_models(models, preprocessor, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates models using weighted F1-score.
    """

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        score = f1_score(y_test, y_pred, average="weighted")

        results[name] = {
            "model": pipeline,
            "f1_score": score
        }

    return results
