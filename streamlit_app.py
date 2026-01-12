import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

from sklearn.metrics import confusion_matrix
from src.preprocessing import preprocess_data
from src.model_selection import get_models, evaluate_models
from src.hyperparameter_tuning import tune_random_forest

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🔍️",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= HEADER =================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
with col2:
    st.title("AutoML Pipeline")
    st.write("An end-to-end automated system for data preprocessing, model selection, and hyperparameter tuning.")

st.divider()

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

optuna_trials = st.sidebar.slider(
    "Optuna Tuning Trials",
    min_value=10,
    max_value=50,
    value=20,
    step=5
)

run_button = st.sidebar.button("🚀 Start AutoML Pipeline")

# ================= MAIN CONTENT =================
if uploaded_file is not None:
    # Data Loading
    os.makedirs("data", exist_ok=True)
    temp_csv_path = "data/temp_dataset.csv"
    df = pd.read_csv(uploaded_file)
    df.to_csv(temp_csv_path, index=False)

    # UI Layout: Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🤖 Model Training", "🏆 Final Results"])

    with tab1:
        st.subheader("Dataset Overview")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", df.shape[0])
        col_b.metric("Columns", df.shape[1])
        col_c.metric("Missing Values", df.isna().sum().sum())

        with st.expander("👀 View Raw Data"):
            st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Target Selection")
        target_column = st.selectbox(
            "Which column do you want to predict?",
            df.columns,
            index=len(df.columns)-1
        )

    with tab2:
        if not run_button:
            st.info("Configure your settings in the sidebar and click 'Run AutoML' to begin.")
        else:
            with st.status("🏗️ Processing Pipeline...", expanded=True) as status:
                st.write("Preprocessing data...")
                X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                    csv_path=temp_csv_path,
                    target_column=target_column
                )
                
                st.write("Evaluating multiple models...")
                models = get_models()
                results = evaluate_models(
                    models, preprocessor, X_train, X_test, y_train, y_test
                )
                
                st.write("Tuning Random Forest hyperparameters...")
                tuned_model, tuned_score = tune_random_forest(
                    preprocessor, X_train, X_test, y_train, y_test
                )
                
                status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

            # Results Display
            st.subheader("Model Comparison")
            # Convert results dict to a cleaner dataframe for display
            res_df = pd.DataFrame(results).T.drop(columns=['model'])
            st.table(res_df.style.highlight_max(axis=0, color='#d4edda'))

            # Logic for Best Model
            best_default_model_name = max(results, key=lambda x: results[x]["f1_score"])
            best_default_model = results[best_default_model_name]["model"]
            best_default_score = results[best_default_model_name]["f1_score"]

            if tuned_score > best_default_score:
                final_model, final_model_name, final_score = tuned_model, "Tuned Random Forest", tuned_score
            else:
                final_model, final_model_name, final_score = best_default_model, best_default_model_name, best_default_score

            os.makedirs("models", exist_ok=True)
            joblib.dump(final_model, "models/best_model.pkl")

    with tab3:
        if run_button:
            st.balloons()
            st.success(f"### 🏆 Winner: {final_model_name}")
            
            c1, c2 = st.columns(2)
            c1.metric("Final F1 Score", f"{final_score:.4f}")
            c2.metric("Tuning Improvement", f"{(tuned_score - best_default_score):.4f}")

            st.divider()
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📈 Confusion Matrix")
                y_pred = final_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            with col_right:
                st.subheader("📊 Feature Correlation")
                numeric_df = df.select_dtypes(include=["number"])
                if not numeric_df.empty:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.info("No numeric columns for correlation.")

            # Download Button
            with open("models/best_model.pkl", "rb") as f:
                st.download_button(
                    label="💾 Download Best Model (.pkl)",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
else:
    st.info("Please upload a CSV file in the sidebar to get started.")