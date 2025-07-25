import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

# --------------------
# Load the model
# --------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_churn_model.cbm")
    return model

model = load_model()

# --------------------
# App title and description
# --------------------
st.title("🔍 Employee Churn Prediction App")
st.write("Predict whether an employee will leave the company based on HR and performance data.")

# --------------------
# Sidebar input fields
# --------------------
st.sidebar.header("Input Employee Features")

satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
last_evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, 0.5, step=0.01)
number_project = st.sidebar.slider("Number of Projects", 1, 10, 3)
average_monthly_hours = st.sidebar.slider("Average Monthly Hours", 100, 400, 200)
time_spend_company = st.sidebar.slider("Years at Company", 1, 10, 3)
work_accident = st.sidebar.radio("Had Work Accident?", [0, 1])
promotion_last_5years = st.sidebar.radio("Promotion in Last 5 Years?", [0, 1])
department = st.sidebar.selectbox("Department", [
    'sales', 'technical', 'support', 'IT', 'hr', 'marketing',
    'product_mng', 'RandD', 'accounting'
])
salary = st.sidebar.selectbox("Salary Level", ['low', 'medium', 'high'])

# --------------------
# Prediction
# --------------------
if st.button("🔮 Predict Churn"):
    # Prepare input in correct column order and names
    input_data = pd.DataFrame([[  
        satisfaction_level,
        last_evaluation,
        number_project,
        average_monthly_hours,  # Will map to the model-expected typo name
        time_spend_company,
        work_accident,
        promotion_last_5years,
        department,
        salary
    ]], columns=[
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',  # ← Match the model's expected column name (with typo)
        'time_spend_company',
        'Work_accident',
        'promotion_last_5years',
        'Department',
        'salary'
    ])

    # Define categorical feature positions
    pool = Pool(input_data, cat_features=[7, 8])  # Department and salary are categorical

    # Predict
    try:
        prediction = model.predict(pool)[0]
        proba = model.predict_proba(pool)[0][1]
    except Exception as e:
        st.error("Prediction failed. Please check model and input format.")
        st.exception(e)
        st.stop()

    # Display result
    if prediction == 1:
        st.error(f"⚠️ Prediction: Employee will **LEAVE** the company (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Prediction: Employee will **STAY** with the company (Confidence: {1 - proba:.2%})")
