import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

st.title("Customer Churn Prediction")

# ---------- Load model & transformers ----------
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# ---------- UI: user inputs ----------
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=100, value=40)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1.0)

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

st.write("### Raw input")
st.json(input_data)


# ---------- Prepare DataFrame ----------
input_df = pd.DataFrame([input_data])   # single-row dataframe

# ---------- Encode Geography (OneHotEncoder expected a 2D input) ----------
geo_encoded = label_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_cols = label_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# ---------- Encode Gender (handle LabelEncoder or OneHotEncoder) ----------
gender_encoded_df = None
try:
    # If encoder is OneHotEncoder (has get_feature_names_out)
    gender_encoded = label_encoder_gender.transform(input_df[['Gender']]).toarray()
    gender_cols = label_encoder_gender.get_feature_names_out(['Gender'])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=gender_cols)
except Exception:
    # Fall back: LabelEncoder (returns 1D array)
    try:
        gender_num = label_encoder_gender.transform(input_df['Gender'])
        # create column name same as original (e.g., 'Gender')
        gender_encoded_df = pd.DataFrame(gender_num, columns=['Gender'])
    except Exception as e:
        st.error(f"Gender encoder error: {e}")
        st.stop()

# ---------- Drop original categorical columns (we'll replace them with encoded) ----------
input_df = input_df.drop(columns=['Geography', 'Gender'])

# ---------- Combine all features in a single DataFrame (order doesn't yet matter) ----------
combined_df = pd.concat(
    [input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True), gender_encoded_df.reset_index(drop=True)],
    axis=1
)

st.write("### Processed features (before scaling)")
st.dataframe(combined_df)

# ---------- Make sure scaler input shape/order matches what scaler expects ----------
# If scaler has feature names recorded, reorder combined_df accordingly
if hasattr(scaler, 'feature_names_in_'):
    scaler_cols = list(getattr(scaler, 'feature_names_in_'))
    # check that all scaler_cols are present
    missing = [c for c in scaler_cols if c not in combined_df.columns]
    if missing:
        st.error(f"Scaler expects these columns but they are missing: {missing}")
        st.stop()
    combined_df = combined_df[scaler_cols]
else:
    # If scaler doesn't store feature names, check numeric match
    if hasattr(scaler, 'n_features_in_'):
        if scaler.n_features_in_ != combined_df.shape[1]:
            st.error(
                f"Scaler expects {scaler.n_features_in_} features but got {combined_df.shape[1]}. "
                "Check column order or how you saved the scaler."
            )
            st.stop()
    # otherwise proceed (best-effort)

# ---------- Scale ----------
try:
    input_data_scaled = scaler.transform(combined_df)
except Exception as e:
    st.error(f"Error when scaling input: {e}")
    st.stop()

# ---------- Predict (button to run) ----------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(np.ravel(prediction)[0])

        st.write("### Prediction Probability")
        st.write(f"{prediction_proba:.4f}")

        if prediction_proba > 0.5:
            st.error(
                f"🔴 The customer is likely to churn.\n\n"
                f"**Churn Probability:** {prediction_proba:.2f}\n\n"
                f"This means the customer may need attention or retention efforts."
            )
        else:
            st.success(
                f"🟢 The customer is unlikely to churn.\n\n"
                f"**Churn Probability:** {prediction_proba:.2f}\n\n"
                f"This means the customer is most likely to stay with the team."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
