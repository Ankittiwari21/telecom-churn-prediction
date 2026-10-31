import streamlit as st
import pandas as pd
import joblib

# 1. Load the frozen model and scaler (adjust path if needed)
# In Streamlit, paths are relative to where you run the command
model = joblib.load('models/churn_rf_model.joblib')
scaler = joblib.load('models/scaler.joblib')

st.title("Telecom Churn Predictor 📉")
st.write("Upload a CSV of customer data to instantly flag high-risk users.")

# 2. Create a file uploader on the web page
uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

if uploaded_file is not None:
    # 3. Read the uploaded data
    input_df = pd.read_csv(uploaded_file)
    st.write("### Raw Customer Data")
    st.dataframe(input_df.head())
    
    if st.button("Predict Churn Risk"):
        with st.spinner("Analyzing customers..."):
            # --- PREPROCESSING (Mirroring Step 3) ---
            # In a true production app, you would import a clean function from src/preprocess.py
            # Here we do a quick version for the demonstration
            if 'customerID' in input_df.columns:
                input_df = input_df.drop('customerID', axis=1)

            if 'Churn' in input_df.columns:
                input_df = input_df.drop('Churn', axis=1)    
                
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
            input_df = input_df.dropna()
            
            # This is where the get_dummies vs OneHotEncoder trap happens! 
            # We force the columns to match exactly what the model trained on.
            df_encoded = pd.get_dummies(input_df, drop_first=True)
            
            # --- SCALING (Mirroring Step 5) ---
            X_scaled = scaler.transform(df_encoded)
            
            # --- PREDICTING ---
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1] # Get % chance of churn
            
            # --- DISPLAY RESULTS ---
            results_df = input_df.copy()
            results_df['Churn_Prediction'] = predictions
            results_df['Risk_Probability'] = probabilities
            
            # Filter to show only the highest risk customers
            high_risk = results_df[results_df['Churn_Prediction'] == 1]
            
            st.error(f"⚠️ Found {len(high_risk)} customers at high risk of churning!")
            st.dataframe(high_risk.sort_values(by='Risk_Probability', ascending=False))