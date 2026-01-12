# 🚀 AutoML Pipeline – End-to-End Automated Machine Learning System

> **Final Year B.Tech (AI & DS) Project**  
> An interactive AutoML system for automated model selection and hyperparameter optimization on tabular classification datasets.

---

## 📌 Overview

This project implements an **end-to-end AutoML (Automated Machine Learning) pipeline** designed to automate repetitive and time-consuming steps in the machine learning workflow.  
Given a tabular dataset, the system automatically performs data preprocessing, trains multiple machine learning models, applies hyperparameter tuning, evaluates performance, and selects the best-performing model.

An **interactive Streamlit web application** is included, allowing users to upload datasets and run the AutoML pipeline without modifying any code.

---

## 🎯 Problem Statement

Traditional machine learning workflows require significant manual effort for:
- Data preprocessing  
- Model selection  
- Hyperparameter tuning  
- Performance evaluation  

This project addresses these challenges by building a **fully automated and reusable AutoML pipeline**, reducing development time while maintaining robust evaluation standards.

---

## ✨ Key Features

- Automated data preprocessing (missing values, encoding, scaling)
- Multiple model training and comparison
- Hyperparameter tuning using **Optuna**
- Automatic best model selection using **weighted F1-score**
- Support for numeric and categorical target labels
- Interactive **Streamlit** web interface
- Model persistence for reuse and inference

---

## 🧠 Supported Learning Task

- Supervised Machine Learning  
- Tabular **Classification**
  - Binary classification
  - Multiclass classification

> ⚠️ Regression, time-series, and unstructured data are not currently supported.

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM)

All models use a shared preprocessing pipeline to ensure fair comparison.

---

## 📊 Evaluation Metric

- **Weighted F1-Score**

Chosen for robustness with class imbalance and categorical target labels.

---

## 🏗️ Project Architecture

```
AutoML-Pipeline
 ┣ 📁 data
 ┣ 📁 src
 ┃ ┣ __init__.py
 ┃ ┣ preprocessing.py
 ┃ ┣ model_selection.py
 ┃ ┣ hyperparameter_tuning.py
 ┃ ┣ evaluation.py
 ┃ ┗ predict.py
 ┣ 📁 models
 ┣ 📄 main.py
 ┣ 📄 streamlit_app.py
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## 🔁 AutoML Workflow

```
Dataset Upload
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Evaluation (F1-Score)
      ↓
Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Model Saved
```

---

## 🖥️ Streamlit Web Application

Features:
- CSV upload
- Target column selection
- One-click AutoML execution
- Visualizations:
  - Dataset preview
  - Correlation heatmap
  - Confusion matrix
- Automatic saving of the best model

---

## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run using CLI
```bash
python main.py
```

### Run Streamlit App
```bash
python -m streamlit run streamlit_app.py
```

App opens at: http://localhost:8501

---

## 💾 Saved Model

```
models/best_model.pkl
```

The saved file contains the full preprocessing + model pipeline.

---

## 🎓 Academic Relevance

- Final Year B.Tech (AI & DS) Project
- Demonstrates AutoML, Optuna, and ML engineering concepts
- Suitable for GitHub, resume, and interviews

---

## 🚀 Future Enhancements

- Auto-detect regression vs classification
- Regression model support
- CLI configuration
- Cloud deployment (Streamlit / FastAPI)

---

## 👨‍💻 Developer

**Kaushik Mane**  
Final Year B.Tech – Artificial Intelligence & Data Science  

- 📧 Email: kaushikmane0708@gmail.com  
- 💼 LinkedIn: https://www.linkedin.com/in/kaushik-mane-806831337  
- 🐙 GitHub: https://github.com/kaushik-0708

---

⭐ If you find this project useful, consider starring the repository!
