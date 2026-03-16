import streamlit as st
import pandas as pd
import joblib

# ====================================
# LOAD MODEL
# ====================================

@st.cache_resource
def load_model():
    model = joblib.load("hotel_cancellation_model.pkl")
    return model

model = load_model()

# ====================================
# PAGE TITLE
# ====================================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
"""
This application predicts whether a hotel booking will be **Canceled or Not Canceled**
using a trained **Machine Learning model**.
"""
)

# ====================================
# BOOKING INFORMATION
# ====================================

st.header("Booking Information")

hotel = st.selectbox(
    "Hotel",
    ["Resort Hotel", "City Hotel"]
)

lead_time = st.number_input(
    "Lead Time (days)",
    min_value=0,
    max_value=365,
    value=30
)

arrival_month = st.selectbox(
    "Arrival Month",
    [
        "1","2","3","4","5","6",
        "7","8","9","10","11","12"
    ]
)

arrival_week = st.slider(
    "Arrival Week Number",
    1, 52, 20
)

arrival_day = st.slider(
    "Arrival Day of Month",
    1, 31, 15
)

# ====================================
# STAY INFORMATION
# ====================================

st.header("Stay Information")

weekend_nights = st.number_input(
    "Weekend Nights",
    min_value=0,
    max_value=10,
    value=1
)

week_nights = st.number_input(
    "Week Nights",
    min_value=0,
    max_value=20,
    value=2
)

# ====================================
# GUEST INFORMATION
# ====================================

st.header("Guest Information")

adults = st.number_input(
    "Adults",
    min_value=1,
    max_value=10,
    value=2
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=5,
    value=0
)

babies = st.number_input(
    "Babies",
    min_value=0,
    max_value=3,
    value=0
)

# ====================================
# CUSTOMER HISTORY
# ====================================

st.header("Customer History")

previous_cancellations = st.number_input(
    "Previous Cancellations",
    min_value=0,
    max_value=10,
    value=0
)

previous_bookings = st.number_input(
    "Previous Bookings Not Canceled",
    min_value=0,
    max_value=20,
    value=0
)

repeated_guest = st.selectbox(
    "Repeated Guest",
    [0,1]
)

# ====================================
# BOOKING DETAILS
# ====================================

st.header("Booking Details")

meal = st.selectbox(
    "Meal",
    ["BB","HB","FB","SC"]
)

market_segment = st.selectbox(
    "Market Segment",
    ["Online TA","Offline TA/TO","Direct","Corporate","Complementary"]
)

distribution_channel = st.selectbox(
    "Distribution Channel",
    ["TA/TO","Direct","Corporate","GDS"]
)

deposit_type = st.selectbox(
    "Deposit Type",
    ["No Deposit","Refundable","Non Refund"]
)

customer_type = st.selectbox(
    "Customer Type",
    ["Transient","Transient-Party","Contract","Group"]
)

# ====================================
# PRICE INFORMATION
# ====================================

st.header("Price Information")

adr = st.number_input(
    "Average Daily Rate (ADR)",
    min_value=0.0,
    value=100.0
)

special_requests = st.number_input(
    "Special Requests",
    min_value=0,
    max_value=5,
    value=0
)

parking = st.number_input(
    "Required Parking Spaces",
    min_value=0,
    max_value=3,
    value=0
)

# ====================================
# PREDICTION
# ====================================

if st.button("Predict Cancellation"):

    # ====================================
    # FEATURE ENGINEERING
    # ====================================

    total_guest = adults + children + babies
    total_nights = weekend_nights + week_nights

    if lead_time <= 7:
        booking_type = "Last Minute"
    elif lead_time <= 30:
        booking_type = "Short Term"
    else:
        booking_type = "Long Term"

    # ====================================
    # CREATE INPUT DATAFRAME
    # ====================================

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

        "reserved_room_type":["A"],
        "assigned_room_type":["A"],

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
        "reservation_status_date":["2024-01-01"],

        "bookingID":[999999]

    })

    # ====================================
    # MODEL PREDICTION
    # ====================================

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ====================================
    # RESULT
    # ====================================

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Booking likely to be **CANCELED**")
    else:
        st.success("✅ Booking likely **NOT canceled**")

    st.write(f"Cancellation Probability: **{probability:.2%}**")