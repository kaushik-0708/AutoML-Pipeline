# main.py

import joblib
from src.preprocessing import preprocess_data
from src.model_selection import get_models, evaluate_models
from src.hyperparameter_tuning import tune_random_forest


def main():
    # Dataset details
    data_path = "data/titanic.csv"
    target_column = "Survived"

    # ---------------- STEP 1: PREPROCESSING ----------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        csv_path=data_path,
        target_column=target_column
    )

    # ---------------- STEP 2: DEFAULT MODEL SELECTION ----------------
    print("\n📊 Running default model selection...\n")

    models = get_models()
    results = evaluate_models(
        models,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_default_model_name = max(
        results, key=lambda x: results[x]["f1_score"]
    )
    best_default_model = results[best_default_model_name]["model"]
    best_default_score = results[best_default_model_name]["f1_score"]

    # ---------------- STEP 3: HYPERPARAMETER TUNING ----------------
    print("\n⚙️ Running hyperparameter tuning (Random Forest)...\n")

    tuned_model, tuned_score = tune_random_forest(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------- STEP 4: FINAL MODEL SELECTION ----------------
    print("\n⚪️ FINAL MODEL RESULT\n")

    if tuned_score > best_default_score:
        joblib.dump(tuned_model, "models/best_model.pkl")
        print("Best Model: Tuned Random Forest")
        print(f"F1 Score: {tuned_score:.4f}")
    else:
        joblib.dump(best_default_model, "models/best_model.pkl")
        print(f"Best Model: {best_default_model_name}")
        print(f"F1 Score: {best_default_score:.4f}")

    print("\n✅ Best model saved successfully at models/best_model.pkl")


if __name__ == "__main__":
    main()
