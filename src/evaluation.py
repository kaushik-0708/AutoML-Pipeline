# src/evaluation.py

import joblib


def select_and_save_best_model(results, save_path="models/best_model.pkl"):
    """
    Selects the best model based on F1-score and saves it.
    """

    # Find model with highest F1-score
    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = results[best_model_name]["model"]
    best_score = results[best_model_name]["f1_score"]

    # Save model
    joblib.dump(best_model, save_path)

    print("\n🏆 Best Model Selected:")
    print(f"Model: {best_model_name}")
    print(f"F1 Score: {best_score:.4f}")
    print(f"Saved at: {save_path}")

    return best_model_name, best_score
