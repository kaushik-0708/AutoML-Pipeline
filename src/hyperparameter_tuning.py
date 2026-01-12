# src/hyperparameter_tuning.py

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


def tune_random_forest(preprocessor, X_train, X_test, y_train, y_test):
    """
    Tunes Random Forest and returns the trained best pipeline.
    """

    def objective(trial):

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        return f1_score(y_test, preds, average="weighted")


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n🔍 Best Hyperparameters Found:")
    print(study.best_params)
    print(f"Best F1 Score: {study.best_value:.4f}")

    # Train final model using best parameters
    best_model = RandomForestClassifier(**study.best_params, random_state=42)

    best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study.best_value
