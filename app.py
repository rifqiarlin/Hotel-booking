import streamlit as st
import pandas as pd
import joblib

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# ====================================
# LOAD MODEL & PREPROCESSOR
# ====================================

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocess.pkl")
    model = joblib.load("models.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()

# ====================================
# TITLE
# ====================================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.caption(
"""
Machine Learning application to predict whether a hotel booking
will be **Canceled or Not Canceled** using a **LightGBM model**.
"""
)

st.divider()

# ====================================
# SIDEBAR INPUT
# ====================================

st.sidebar.header("📋 Booking Input")

# Booking Info
st.sidebar.subheader("Booking Information")

hotel = st.sidebar.selectbox(
    "Hotel",
    [
        'Resort Hotel',
        'City Hotel'
    ]
)

lead_time = st.sidebar.number_input(
    "Lead Time (days)",
    0,365,30
)

arrival_month = st.sidebar.selectbox(
    "Arrival Month",
    list(range(1,13))
)

arrival_week = st.sidebar.slider(
    "Arrival Week Number",
    1,52,20
)

arrival_day = st.sidebar.slider(
    "Arrival Day of Month",
    1,31,15
)

# Stay Info
st.sidebar.subheader("Stay Information")

weekend_nights = st.sidebar.number_input(
    "Weekend Nights",
    0,10,1
)

week_nights = st.sidebar.number_input(
    "Week Nights",
    0,20,2
)

# Guest Info
st.sidebar.subheader("Guest Information")

adults = st.sidebar.number_input("Adults",1,10,2)
children = st.sidebar.number_input("Children",0,5,0)
babies = st.sidebar.number_input("Babies",0,3,0)

# Customer History
st.sidebar.subheader("Customer History")

previous_cancellations = st.sidebar.number_input(
    "Previous Cancellations",
    0,10,0
)

previous_bookings = st.sidebar.number_input(
    "Previous Bookings Not Canceled",
    0,20,0
)

repeated_guest = st.sidebar.selectbox(
    "Repeated Guest",
    [0,1]
)

# Booking Details
st.sidebar.subheader("Booking Details")

meal = st.sidebar.selectbox(
    "Meal",
    ["BB","HB","FB","SC"]
)

market_segment = st.sidebar.selectbox(
    "Market Segment",
    ["Online TA","Offline TA/TO","Direct","Corporate"]
)

distribution_channel = st.sidebar.selectbox(
    "Distribution Channel",
    ["TA/TO","Direct","Corporate","GDS"]
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    ["No Deposit","Refundable","Non Refund"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    ["Transient","Transient-Party","Contract","Group"]
)

reserved_room_type = st.sidebar.selectbox(
    "Reserved Room Type",
    ["A","B","C","D","E","F","G"]
)

# Price Info
st.sidebar.subheader("Price Information")

adr = st.sidebar.number_input(
    "Average Daily Rate (ADR)",
    0.0,300.0,100.0
)

special_requests = st.sidebar.number_input(
    "Special Requests",
    0,5,0
)

parking = st.sidebar.number_input(
    "Required Parking Spaces",
    0,3,0
)

predict_button = st.sidebar.button("Predict Cancellation")

# ====================================
# MAIN DASHBOARD
# ====================================

st.header("Prediction Dashboard")

if predict_button:

    # Feature Engineering
    total_guest = adults + children + babies
    total_nights = weekend_nights + week_nights

    if lead_time <= 7:
        booking_type = "Last Minute"
    elif lead_time <= 30:
        booking_type = "Short Term"
    else:
        booking_type = "Long Term"

    # Input DataFrame
    input_data = pd.DataFrame({

        "hotel":[hotel],
        "lead_time":[lead_time],
        "arrival_date_year":[2024],
        "arrival_date_month":[arrival_month],
        "arrival_date_week_number":[arrival_week],
        "arrival_date_day_of_month":[arrival_day],

        "stays_in_weekend_nights":[weekend_nights],
        "stays_in_week_nights":[week_nights],

        "adults":[adults],
        "children":[children],
        "babies":[babies],

        "total_guest":[total_guest],
        "total_nights":[total_nights],
        "booking_type":[booking_type],

        "meal":[meal],
        "country":["PRT"],

        "market_segment":[market_segment],
        "distribution_channel":[distribution_channel],

        "is_repeated_guest":[repeated_guest],

        "previous_cancellations":[previous_cancellations],
        "previous_bookings_not_canceled":[previous_bookings],

        "reserved_room_type":[reserved_room_type],
        "assigned_room_type":[reserved_room_type],

        "booking_changes":[0],

        "deposit_type":[deposit_type],

        "agent":[1],
        "company":[0],

        "days_in_waiting_list":[0],

        "customer_type":[customer_type],

        "adr":[adr],

        "required_car_parking_spaces":[parking],

        "total_of_special_requests":[special_requests],

        "reservation_status":["Check-Out"],
        "reservation_status_date":["2024-01-01"]
    })

    # ====================================
    # PREPROCESSING
    # ====================================

    X = preprocessor.transform(input_data)

    # ====================================
    # PREDICTION
    # ====================================
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0][1]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:
            st.error("⚠️ High Risk of Cancellation")
        else:
            st.success("✅ Booking is Safe")

    with col2:

        st.metric(
            "Cancellation Probability",
            f"{probability:.2%}"
        )

    st.write("### Probability Level")

    st.progress(probability)

    st.write(
        f"The model predicts **{probability:.2%} probability** that this booking will be canceled."
    )

# ====================================
# MODEL INFO
# ====================================

st.divider()

st.markdown("### 📊 Model Information")

st.info(
"""
This application uses a **LightGBM Machine Learning model** 
that has been **hyperparameter tuned** to optimize predictive performance.

**Project Details**

- Algorithm : LightGBM  
- Model Type : Binary Classification  
- Optimization : Hyperparameter Tuning  
- Development Year : **2026**  
- Use Case : Hotel Booking Cancellation Prediction
"""
)