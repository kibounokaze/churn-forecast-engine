# 📊 Customer Churn Prediction

A machine learning project aimed at predicting customer **churn** using classical ML models, exploratory data analysis, preprocessing pipelines, and model comparison experiments.


## 🚀 Project Overview

Customer churn prediction is a critical problem for businesses, especially subscription- and service-based companies.

In this project, we explore different machine learning techniques to identify customers who are likely to churn and require proactive engagement.

This repository includes:

* Clean project structure following best practices
* Clear separation between raw and processed data
* Fully reproducible notebooks
* A baseline comparison of **Logistic Regression**, **Random Forest**, and **SVC**
* A focus on handling class imbalance
* Selection of **SVC** as the final model based on **recall performance for churners**

## 📚 Data Source and Attribution

The data used in this project is the **[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)**, originally hosted on Kaggle.

## 🎯 Goal of the Project

Our main objective is to **detect churners as accurately as possible**.

In churn prediction:

* Missing a churner (**false negative**) is costly
* We prefer a model that captures the majority of actual churners
* **Recall** and **F1-score** for the minority class are more important than accuracy

After comparing multiple models, **SVC demonstrated the strongest recall and F1-score for churners**, making it the most suitable choice for this business scenario.

## 📁 Repository Structure

* `project-root/`
    * `app/`
        * `app.py` # Streamlit app
    * `data/`
        * `raw/` # Original dataset(s)
        * `processed/` # Cleaned, transformed, feature-engineered data
    * `models/`
        * `model.pkl` # Final saved model
        * `scaler.pkl` # Scaler
    * `notebooks/`
        * `data_exploration.ipynb` # EDA
        * `data_preprocessing.ipynb` # Preprocessing
        * `model_building.ipynb` # Model training
    * `requirements.txt` # Python dependencies
    * `README.md` # Project documentation

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

The EDA notebook covers:

* Distribution of numerical and categorical features
* Missing value analysis
* Outliers and correlations
* Class imbalance visualization
* Behavioral insights and patterns linked to churn

### 2. Data Preprocessing

Performed in `data_preprocessing.ipynb`, including:

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Train/test split
* Dealing with class imbalance (class weights)
* Saving processed datasets for reproducibility

> Processed data is stored under: `/data/processed`

### 3. Model Training & Evaluation

Implemented models include:

* Logistic Regression
* Random Forest Classifier
* Support Vector Classifier (SVC)

Each model was evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* Class-specific performance (especially for churners)


## 🧠 Why We Chose SVC as the Final Model

Churn datasets are typically **imbalanced** (majority non-churners).

In this scenario:

* Accuracy becomes misleading
* The focus shifts to the positive class (**churners**)
* Missing churners introduces **high business risk**

| Model | Churn Recall | Churn Precision | Churn F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | Moderate | Low | Low | Moderate |
| Random Forest | Low | High | Moderate | High |
| **SVC** | **Highest** | Moderate | **Highest** | **Highest** |

➡️ **SVC provided the best recall and F1-score for churners**, meaning it identifies the highest number of true churners.

This aligns with the project objective: **prioritize detecting churners even if it introduces slightly more false positives.**

Thus, **SVC was selected as the recommended model**.


## 🎥 Demo Video

https://github.com/user-attachments/assets/1a0ca2b0-cf90-475c-b2fe-c24147920cce

## 🔧 Setup

1.  Create a virtual environment

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate     # Windows
    ```

2.  Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

3.  Run Streamlit Interface

    ```bash
    streamlit run app/app.py
    ```
