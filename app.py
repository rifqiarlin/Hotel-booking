import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hotel Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    data = joblib.load("hotel_cancellation_model.pkl")
    return data["model"], data["features"]

model, feature_names = load_model()

# =========================
# HEADER
# =========================
st.title("🏨 Hotel Booking Cancellation Prediction") 
st.markdown( """ Machine Learning dashboard to predict **hotel booking cancellations** and analyze risk factors using a **LightGBM classification model**. """ )
st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Booking Information")

lead_time = st.sidebar.slider("Lead Time (days before arrival)", 0, 447, 0)

adr = st.sidebar.number_input(
    "ADR (Average Daily Rate)",
    min_value=0.0,
    max_value=252.0,
    value=0.0
)

total_nights = st.sidebar.slider("Total Nights", 1, 69, 1)

total_guests = st.sidebar.slider("Total Guests", 1, 55, 1)

previous_cancellations = st.sidebar.slider(
    "Previous Cancellations",
    0, 26, 0
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

meal = st.sidebar.selectbox(
    "Meal",
    ["BB", "HB", "FB", "SC", "Undefined"]
)

booking_type = st.sidebar.selectbox(
    "Booking Channel",
    ["Direct", "Corporate", "Online Travel Agent"]
)

reserved_room_type = st.sidebar.selectbox(
    "Reserved Room Type",
    ["A","B","C","D","E","F","G","H","L"]
)

predict = st.sidebar.button("Predict")

# =========================
# MAIN PAGE
# =========================
col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("Booking Prediction")

with col_right:
    st.info("Enter booking information in the sidebar to generate prediction.")

# =========================
# PREDICTION
# =========================
if predict:

    with st.spinner("Running prediction model..."):

        input_data = {
            "lead_time": lead_time,
            "adr": adr,
            "total_nights": total_nights,
            "total_guests": total_guests,
            "previous_cancellations": previous_cancellations,
            "deposit_type": deposit_type,
            "customer_type": customer_type,
            "meal": meal,
            "booking_type": booking_type,
            "reserved_room_type": reserved_room_type
        }

        input_df = pd.DataFrame([input_data])

        # Encoding
        input_df = pd.get_dummies(input_df)

        # Align columns
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.divider()

    # =========================
    # RISK LEVEL
    # =========================
    if probability < 0.3:
        risk = "Low Risk"
        color = "green"
    elif probability < 0.7:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cancellation Probability", f"{probability:.2%}")

    with col2:
        if prediction == 1:
            st.metric("Prediction Result", "Canceled ❌")
        else:
            st.metric("Prediction Result", "Not Canceled ✅")

    with col3:
        st.metric("Risk Level", risk)

    # =========================
    # PROBABILITY BAR
    # =========================
    st.subheader("Cancellation Risk Level")

    st.progress(probability)

    # =========================
    # BOOKING SUMMARY
    # =========================
    st.subheader("Booking Summary")

    summary_df = pd.DataFrame([input_data])

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    st.success("Prediction completed successfully!")

st.divider()

st.markdown("### 📊 Model Details")

st.markdown(
"""
This application utilizes a **Light Gradient Boosting Machine (LightGBM)** model
that has been carefully **tuned using hyperparameter optimization techniques**
to improve predictive performance.

The model is designed to estimate the probability of **hotel booking cancellations**
based on booking characteristics, guest information, and reservation details.

**Project Information**

- **Algorithm:** LightGBM  
- **Model Type:** Binary Classification  
- **Optimization:** Hyperparameter Tuning  
- **Development Year:** 2026  
- **Objective:** Predict hotel booking cancellation risk
"""
)