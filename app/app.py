import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datetime import datetime

# ----------------------------- Load Environment Variables -----------------------------
load_dotenv()

# Personal & Branding
YOUR_NAME = os.getenv("YOUR_NAME")
YOUR_LINKEDIN = os.getenv("YOUR_LINKEDIN")
YOUR_GITHUB = os.getenv("YOUR_GITHUB")
YOUR_PORTFOLIO = os.getenv("YOUR_PORTFOLIO")
PROJECT_DATE = os.getenv("PROJECT_DATE")
DASHBOARD_TITLE = os.getenv("DASHBOARD_TITLE")
DASHBOARD_DESC = os.getenv("DASHBOARD_DESC")

# Page Headers
PAGE_SINGLE_HEADER = os.getenv("PAGE_SINGLE_HEADER")
PAGE_BATCH_HEADER = os.getenv("PAGE_BATCH_HEADER")
PAGE_OVERVIEW_HEADER = os.getenv("PAGE_OVERVIEW_HEADER")
PAGE_MODEL_HEADER = os.getenv("PAGE_MODEL_HEADER")

# Info & Captions
BATCH_INFO = os.getenv("BATCH_INFO")
SAMPLE_TEMPLATE_HELP = os.getenv("SAMPLE_TEMPLATE_HELP")
HIGH_RISK_TITLE = os.getenv("HIGH_RISK_TITLE")
HIGH_RISK_CAPTION = os.getenv("HIGH_RISK_CAPTION")
ROI_TITLE = os.getenv("ROI_TITLE")
ROI_INPUT_LABEL = os.getenv("ROI_INPUT_LABEL")
ROI_SLIDER_LABEL = os.getenv("ROI_SLIDER_LABEL")
ROI_CURRENT_LOSS = os.getenv("ROI_CURRENT_LOSS")
ROI_SAVINGS_MSG = os.getenv("ROI_SAVINGS_MSG")

# Model Rationale
MODEL_LOGREG_RECALL = os.getenv("MODEL_LOGREG_RECALL")
MODEL_RF_RECALL = os.getenv("MODEL_RF_RECALL")
MODEL_SVC_RECALL = os.getenv("MODEL_SVC_RECALL")
MODEL_SELECTION_CAPTION = os.getenv("MODEL_SELECTION_CAPTION")

# Why SVC
MODEL_WHY_TITLE = os.getenv("MODEL_WHY_TITLE")
MODEL_WHY_TEXT = os.getenv("MODEL_WHY_TEXT", "- **Business Priority**...")  # Multi-line preserved

# Feature Importance
MODEL_FEATURE_TITLE = os.getenv("MODEL_FEATURE_TITLE")

# Risk Drivers
RISK_DRIVER_AGE = os.getenv("RISK_DRIVER_AGE")
RISK_DRIVER_INACTIVE = os.getenv("RISK_DRIVER_INACTIVE")
RISK_DRIVER_PRODUCTS = os.getenv("RISK_DRIVER_PRODUCTS")
RISK_DRIVER_GERMANY = os.getenv("RISK_DRIVER_GERMANY")
RISK_DRIVER_HIGH_BALANCE = os.getenv("RISK_DRIVER_HIGH_BALANCE")
RISK_LOW_PROFILE = os.getenv("RISK_LOW_PROFILE")

# Actions
ACTION_HIGH = os.getenv("ACTION_HIGH")
ACTION_MEDIUM = os.getenv("ACTION_MEDIUM" )
ACTION_LOW = os.getenv("ACTION_LOW")

# ----------------------------- Load Artifacts -----------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    data_path = os.path.join(base_dir, "data", "raw", "churn_data.csv")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df_raw = pd.read_csv(data_path)
    return model, scaler, df_raw

model, scaler, df = load_artifacts()
df = df.drop('customer_id', axis=1)

# ----------------------------- Page Config & Style -----------------------------
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e293b; }
    .stButton>button {
        background-color: #4f46e5; color: white; border-radius: 8px; font-weight: 600; width: 100%;
    }
    .risk-low { color: #16a34a; font-weight: bold; }
    .risk-medium { color: #ca8a04; font-weight: bold; }
    .risk-high { color: #dc2626; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title(f"🏦 {DASHBOARD_TITLE}")
st.markdown(f"**{DASHBOARD_DESC}** • Built by **{YOUR_NAME}** • {PROJECT_DATE}")

# ----------------------------- Sidebar Navigation & About -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
    ["🔮 Single Customer Prediction", 
     "📁 Batch Prediction",
     "📊 Business Overview", 
     "🧠 Model Insights"])

st.sidebar.markdown("---")
st.sidebar.subheader("About the Author")
st.sidebar.markdown(f"**{YOUR_NAME}**")
st.sidebar.caption("Data Scientist | Machine Learning Enthusiast")

if YOUR_LINKEDIN:
    st.sidebar.markdown(f"[LinkedIn]({YOUR_LINKEDIN})")
if YOUR_GITHUB:
    st.sidebar.markdown(f"[GitHub]({YOUR_GITHUB})")
if YOUR_PORTFOLIO:
    st.sidebar.markdown(f"[Portfolio]({YOUR_PORTFOLIO})")

st.sidebar.markdown("---")
st.sidebar.caption("Data: Bank Customer Churn Dataset (Kaggle) • Optimized for detecting churners")

# ============================== Single Customer Prediction ==============================
if page == "🔮 Single Customer Prediction":
    st.header(PAGE_SINGLE_HEADER)
    
    with st.form("single_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            age = st.slider("Age", 18, 92, 38)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance (€)", 0.0, 250000.0, 0.0, step=1000.0)
        with col2:
            estimated_salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 50000.0, step=1000.0)
            products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
            gender = st.selectbox("Gender", ["Male", "Female"])
            country = st.selectbox("Country", ["France", "Spain", "Germany"])
        
        col3, col4 = st.columns(2)
        with col3:
            has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        with col4:
            is_active_member = st.selectbox("Active Member?", ["Yes", "No"])
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")
        
        if submitted:
            input_data = {
                "credit_score": credit_score,
                "gender": 1 if gender == "Male" else 0,
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products_number,
                "credit_card": 1 if has_credit_card == "Yes" else 0,
                "active_member": 1 if is_active_member == "Yes" else 0,
                "estimated_salary": estimated_salary,
                "country_Germany": 1 if country == "Germany" else 0,
                "country_Spain": 1 if country == "Spain" else 0,
            }
            df_input = pd.DataFrame([input_data])
            num_features = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
            df_input[num_features] = scaler.transform(df_input[num_features])
            
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Prediction", "Yes" if prediction else "No")
            with col2:
                st.metric("Churn Probability", f"{probability:.1%}")
            with col3:
                if probability < 0.35:
                    risk, icon, color = "Low Risk", "✅", "risk-low"
                elif probability < 0.65:
                    risk, icon, color = "Medium Risk", "⚠️", "risk-medium"
                else:
                    risk, icon, color = "High Risk", "🚨", "risk-high"
                st.markdown(f"<div class='{color}' style='font-size:1.6rem;text-align:center'>{icon}<br>{risk}</div>", unsafe_allow_html=True)
            
            st.markdown("### Key Risk Drivers")
            risks = []
            if age > 45: risks.append(RISK_DRIVER_AGE)
            if is_active_member == "No": risks.append(RISK_DRIVER_INACTIVE)
            if products_number >= 3: risks.append(RISK_DRIVER_PRODUCTS)
            if country == "Germany": risks.append(RISK_DRIVER_GERMANY)
            if balance > 100000 and is_active_member == "No": risks.append(RISK_DRIVER_HIGH_BALANCE)
            if not risks: risks.append(RISK_LOW_PROFILE)
            for r in risks: st.markdown(f"• {r}")
            
            st.markdown("### Recommended Action")
            if probability > 0.65:
                st.error(ACTION_HIGH)
            elif probability > 0.35:
                st.warning(ACTION_MEDIUM)
            else:
                st.success(ACTION_LOW)
    
    if 'prediction' in locals():
        result_df = pd.DataFrame([{
            "credit_score": credit_score, "age": age, "tenure": tenure, "balance": balance,
            "products_number": products_number, "estimated_salary": estimated_salary,
            "country": country, "gender": gender, "credit_card": has_credit_card,
            "active_member": is_active_member, "churn_prediction": "Yes" if prediction else "No",
            "churn_probability": round(probability, 4), "risk_level": risk
        }])
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download This Prediction",
            data=csv,
            file_name=f"single_churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ============================== Batch Prediction ==============================
elif page == "📁 Batch Prediction":
    st.header(PAGE_BATCH_HEADER)
    st.info(BATCH_INFO)
    
    sample_data = pd.DataFrame([
        {"credit_score": 650, "age": 42, "tenure": 3, "balance": 75000.0, "products_number": 2, "estimated_salary": 80000.0,
         "gender": "Male", "country": "Germany", "credit_card": "Yes", "active_member": "No"},
        {"credit_score": 720, "age": 35, "tenure": 8, "balance": 0.0, "products_number": 1, "estimated_salary": 120000.0,
         "gender": "Female", "country": "France", "credit_card": "Yes", "active_member": "Yes"},
        {"credit_score": 550, "age": 55, "tenure": 1, "balance": 150000.0, "products_number": 3, "estimated_salary": 60000.0,
         "gender": "Male", "country": "Spain", "credit_card": "No", "active_member": "No"}
    ])
    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=sample_csv,
        file_name="sample_batch_churn_input.csv",
        mime="text/csv",
        help=SAMPLE_TEMPLATE_HELP
    )
    
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(batch_df)} records")
        st.dataframe(batch_df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing..."):
                input_df = batch_df.copy()
                input_df.columns = input_df.columns.str.lower().str.replace(' ', '_').str.strip()
                
                input_df['gender'] = input_df['gender'].astype(str).str.lower().map({"male": 1, "female": 0})
                
                yes_no_map = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}
                if 'credit_card' in input_df.columns:
                    input_df['credit_card'] = input_df['credit_card'].astype(str).str.lower().map(yes_no_map)
                if 'active_member' in input_df.columns:
                    input_df['active_member'] = input_df['active_member'].astype(str).str.lower().map(yes_no_map)
                
                input_df['country'] = input_df['country'].astype(str).str.capitalize()
                input_df = pd.get_dummies(input_df, columns=['country'])
                input_df.rename(columns={
                    'country_germany': 'country_Germany',
                    'country_spain': 'country_Spain'
                }, inplace=True)
                
                required_cols = ['credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
                                 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain']
                for col in required_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[required_cols]
                num_features = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
                input_df[num_features] = scaler.transform(input_df[num_features])
                
                predictions = model.predict(input_df)
                probabilities = model.predict_proba(input_df)[:, 1]
                
                result_df = batch_df.copy()
                result_df['churn_prediction'] = predictions
                result_df['churn_probability'] = probabilities
                result_df['risk_level'] = result_df['churn_probability'].apply(
                    lambda p: "High" if p >= 0.65 else "Medium" if p >= 0.35 else "Low"
                )
            
            st.success(f"Batch prediction complete for {len(result_df)} customers!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk", (result_df['risk_level'] == 'High').sum())
            col2.metric("Medium Risk", (result_df['risk_level'] == 'Medium').sum())
            col3.metric("Low Risk", (result_df['risk_level'] == 'Low').sum())
            
            st.dataframe(result_df[['churn_prediction', 'churn_probability', 'risk_level']].head(10))
            
            output_csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Results",
                output_csv,
                f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )

# ============================== Business Overview ==============================
elif page == "📊 Business Overview":
    st.header(PAGE_OVERVIEW_HEADER)
    
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    churned = df['churn'].sum()
    churn_rate = df['churn'].mean()
    avg_salary = df['estimated_salary'].mean()
    
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    col4.metric("Avg Salary", f"€{avg_salary:,.0f}")
    
    st.markdown("---")
    
    st.subheader("Model Selection Rationale")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logistic Regression", f"Recall: {MODEL_LOGREG_RECALL}", delta="Moderate")
    with col2:
        st.metric("Random Forest", f"Recall: {MODEL_RF_RECALL}", delta="Misses many churners")
    with col3:
        st.metric("**SVC (Selected)**", f"Recall: {MODEL_SVC_RECALL}", delta="Best churn detection")
    st.caption(MODEL_SELECTION_CAPTION)
    
    st.markdown("---")
    
    st.subheader(HIGH_RISK_TITLE)
    high_risk = df[
        (df['active_member'] == 0) &
        (df['age'] > 45) &
        (df['products_number'] >= 3)
    ].nlargest(10, 'balance')
    
    st.dataframe(high_risk[['age', 'balance', 'products_number', 'active_member', 'country', 'estimated_salary', 'churn']])
    st.caption(HIGH_RISK_CAPTION)
    
    st.markdown("---")
    
    st.subheader(ROI_TITLE)
    col1, col2 = st.columns(2)
    with col1:
        avg_customer_value = st.number_input(ROI_INPUT_LABEL, 1000, 100000, 10000)
    with col2:
        potential_churn_reduction = st.slider(ROI_SLIDER_LABEL, 5, 50, 20)
    
    current_annual_loss = churned * avg_customer_value
    projected_savings = current_annual_loss * (potential_churn_reduction / 100)
    
    st.metric(ROI_CURRENT_LOSS, f"€{current_annual_loss:,.0f}")
    st.success(ROI_SAVINGS_MSG.format(reduction=potential_churn_reduction, savings=int(projected_savings)))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        df['churn'].value_counts().plot.pie(labels=['Stayed', 'Churned'], autopct='%1.1f%%', colors=['#86efac', '#fca5a5'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Churn Rate by Country")
        fig, ax = plt.subplots()
        df.groupby('country')['churn'].mean().plot(kind='bar', color=['#93c5fd', '#fdba74', '#fca5a5'], ax=ax)
        ax.set_ylabel('Churn Rate')
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age vs Balance (colored by Churn)")
        sample_df = df.sample(min(2000, len(df)))
        fig = sns.scatterplot(data=sample_df, x='age', y='balance', hue='churn', palette='Set1', alpha=0.6)
        st.pyplot(fig.figure)
    
    with col2:
        st.subheader("Churn Rate by Products & Activity")
        fig = sns.barplot(data=df, x='products_number', y='churn', hue='active_member', palette='Set2')
        plt.ylabel('Churn Rate')
        st.pyplot(fig.figure)

# ============================== Model Insights ==============================
else:
    st.header(PAGE_MODEL_HEADER)
    
    st.subheader(MODEL_WHY_TITLE)
    st.markdown(MODEL_WHY_TEXT)
    
    st.subheader(MODEL_FEATURE_TITLE)
    importance = [
        ('active_member', 0.22), ('age', 0.18), ('products_number', 0.14),
        ('country_Germany', 0.10), ('balance', 0.08), ('credit_score', 0.05),
        ('tenure', 0.03), ('estimated_salary', 0.02)
    ]
    imp_df = pd.DataFrame(importance, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Permutation Importance (Test Set)")
    st.pyplot(fig)