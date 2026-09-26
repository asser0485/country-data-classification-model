
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Country Cluster Prediction')

# Load the trained model
model = joblib.load('model.joblib')

# Load the original data to get feature names and potentially min/max values for sliders
try:
    df_original = pd.read_csv('clustered_country_data.csv')
    features = df_original.drop(columns=df_original.columns[-1]).columns.tolist() # Exclude the last (target) column
except FileNotFoundError:
    st.error("Original data file 'clustered_country_data.csv' not found. Please ensure it's in the same directory.")
    st.stop()

st.sidebar.header('Input Features')

# Create input widgets for each feature dynamically
input_data = {}
for feature in features:
    # Attempt to get min/max for numerical inputs, default to generic range if not found or if object type
    if df_original[feature].dtype in [np.float64, np.int64]:
        min_val = float(df_original[feature].min())
        max_val = float(df_original[feature].max())
        default_val = float(df_original[feature].mean())
        input_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()}', min_val, max_val, default_val)
    else:
        # For other types, a text input might be more appropriate, or specific handling
        input_data[feature] = st.sidebar.text_input(f'{feature.replace("_", " ").title()}', '0')
        try:
            input_data[feature] = float(input_data[feature]) # Try converting to float
        except ValueError:
            st.sidebar.warning(f"Please enter a numerical value for {feature}")
            input_data[feature] = 0.0 # Default to 0 if conversion fails


# Convert input data to DataFrame for prediction
input_df = pd.DataFrame([input_data])

st.subheader('Input Data')
st.write(input_df)

if st.button('Predict Cluster'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Cluster')
    st.success(f'The predicted cluster for the given inputs is: **{prediction[0]}**')
