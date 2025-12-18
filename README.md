# RetentionAI: Churn Forecast Engine & Risk Intelligence

**A production-ready customer churn prediction system** built with classical ML, featuring interactive single & batch inference, business insights, retention ROI calculator, and explainable predictions.

**Live Demo**: 🔗 [https://churn-forecast-engine-demo-pubp.streamlit.app/](https://churn-forecast-engine-demo-pubp.streamlit.app/)

**RetentionAI Demo Video**: 🔗 [https://github.com/kibounokaze/churn-forecast-engine/blob/main/assets/raw/demo-file-video.mp4](https://github.com/kibounokaze/churn-forecast-engine/blob/main/assets/raw/demo-file-video.mp4)

*(Click to watch a full walkthrough of the dashboard — single prediction, batch processing, ROI calculator, and model insights)*

## 🚀 Project Overview

Customer churn is one of the most critical challenges for subscription-based businesses. Early detection enables proactive retention, significantly reducing revenue loss.

**RetentionAI** is an end-to-end ML solution that:
- Predicts churn risk at individual and batch level
- Provides actionable insights and risk drivers
- Quantifies business impact through ROI estimation
- Offers full interpretability for stakeholder trust

Built with a focus on **real-world deployment** and **business impact**, this project demonstrates strong data science and ML engineering skills.

## 📊 Key Results (Test Set)

| Model                  | Churn Recall | Churn Precision | Churn F1 | ROC-AUC  |
|------------------------|--------------|-----------------|----------|----------|
| Logistic Regression    | ~70%         | Moderate        | Moderate | ~0.83    |
| Random Forest          | ~44%         | High            | Moderate | ~0.86    |
| **SVC (Selected)**     | **~74%**     | Moderate        | **Best** | **0.859**|

**Why SVC Was Selected**  

In churn prediction, **false negatives (missing a churner)** are far more expensive than false positives.  
SVC with class weights and RBF kernel delivered the **highest recall** on the minority class while maintaining strong discrimination — making it the optimal choice for minimizing revenue loss.

## 🎯 Core Features

### 🔮 Single Customer Prediction

- Input customer profile via intuitive form
- Instant churn probability & risk level (Low/Medium/High)
- Personalized risk drivers (e.g., "Inactive member", "Older age", "Germany resident")
- Recommended retention actions
- Downloadable prediction report

### 📁 Batch Prediction
- Upload CSV with thousands of customers
- Robust preprocessing (case-insensitive columns, flexible Yes/No handling)
- Risk stratification (High/Medium/Low)
- Summary metrics + full results download

### 📊 Business Overview
- Key metrics: Total customers, churn rate, average salary
- Interactive visualizations (churn distribution, country-wise rates, age vs balance)
- Model comparison card explaining selection rationale
- **Top 10 High-Risk Customer Profiles** — priority targets for retention campaigns
- **Retention ROI Calculator** — estimate annual savings from reducing churn

### 🧠 Model Insights
- Detailed rationale for choosing SVC over alternatives
- Permutation feature importance (top drivers: activity, age, products, country)
- Transparent evaluation metrics

## 📁 Repository Structure

```bash
churn-forecast-engine/
├── app/
│   └── app.py                      # Streamlit dashboard
├── data/
│   ├── raw/churn_data.csv          # Original Kaggle dataset
│   └── processed/                  # Train/test splits & scaled data
├── models/
│   ├── model.pkl                   # Tuned SVC model
│   └── scaler.pkl                  # Feature scaler
├── notebooks/
│   ├── data_exploration.ipynb      # Comprehensive EDA
│   ├── data_preprocessing.ipynb    # Encoding, scaling, splitting
│   └── model_building.ipynb        # Model comparison & tuning
├── requirements.txt
├── .env.example                    # Configuration template
└── README.md
```

## 🔧 Tech Stack & Tools

- **Python 3.10+**
- **scikit-learn**: SVC, GridSearchCV, permutation importance
- **Streamlit**: Interactive dashboard with batch processing
- **Pandas, Matplotlib, Seaborn**: Data processing & visualization
- **python-dotenv**: Modular, secure configuration
- **Streamlit Community Cloud**: Free deployment with secrets management

## 🛠 Challenges & Solutions

- **Class Imbalance (~20% churn)** → Used class weights instead of resampling to preserve data integrity
- **High Recall Requirement** → Prioritized recall over precision based on business cost analysis
- **Batch Inference Robustness** → Built flexible preprocessing to handle varied input formats
- **Interpretability** → Added permutation importance and per-prediction explanations
- **Deployment Configuration** → Used environment variables and Streamlit secrets for clean, secure setup

## 🚀 Future Improvements (Potential Extensions)

- Integrate XGBoost or LightGBM for comparison
- Add probability calibration for better risk scoring
- Connect to real-time database/API
- Add customer segmentation (clustering)
- Email/SMS alert integration for high-risk predictions

## 🛠 Local Setup

```bash
git clone https://github.com/kibounokaze/churn-forecast-engine.git
cd churn-forecast-engine

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app/app.py
```

🌐 Deployment

Deployed on Streamlit Community Cloud using GitHub integration and secrets management.

👨‍💻 Author

Piyush Patil
Data Scientist | Machine Learning Engineer
Built December 2025

Thank you for exploring RetentionAI!
This project showcases full-cycle ML engineering: from data to deployed, business-impactful application.

⭐ Star the repo if you found it valuable!

Feel free to fork and extend it.
Feedback and contributions welcome!