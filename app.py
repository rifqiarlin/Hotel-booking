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
# LOAD MODEL
# ====================================

@st.cache_resource
def load_model():
    return joblib.load("hotel_cancellation_model.pkl")

model = load_model()

# ====================================
# TITLE
# ====================================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write("""
Predict whether a **hotel booking will be canceled or not** using a  
**LightGBM Machine Learning Model**.

Model: **LightGBM (Tuned)**  
Project Year: **2026**
""")

st.divider()

# ====================================
# USER INPUT
# ====================================

st.header("Booking Information")

col1, col2, col3 = st.columns(3)

with col1:
    hotel = st.selectbox("Hotel", ["Resort Hotel", "City Hotel"])
    lead_time = st.number_input("Lead Time", 0, 800, 50)
    arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017])

    arrival_date_month = st.selectbox(
        "Arrival Month",
        ["1","2","3","4","5","6",
         "7","8","9","10","11","12"]
    )

    arrival_date_week_number = st.number_input("Arrival Week Number", 1, 53, 30)
    arrival_date_day_of_month = st.number_input("Arrival Day", 1, 31, 15)

with col2:
    stays_in_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    stays_in_week_nights = st.number_input("Week Nights", 0, 20, 3)

    adults = st.number_input("Adults", 1, 10, 2)
    children = st.number_input("Children", 0, 5, 0)
    babies = st.number_input("Babies", 0, 5, 0)

    country = st.text_input("Country Code", "PRT")

with col3:
    meal = st.selectbox("Meal", ["BB","HB","FB","SC"])

    market_segment = st.selectbox(
        "Market Segment",
        ["Online TA","Offline TA/TO","Direct","Groups","Corporate","Complementary","Aviation"]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO","Direct","Corporate","GDS"]
    )

    is_repeated_guest = st.selectbox("Repeated Guest", [0,1])

st.divider()

# ====================================
# BOOKING HISTORY
# ====================================

st.header("Booking History")

col4, col5, col6 = st.columns(3)

with col4:
    previous_cancellations = st.number_input("Previous Cancellations",0,10,0)
    previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings",0,20,0)

with col5:
    reserved_room_type = st.selectbox("Reserved Room Type", list("ABCDEFGH"))
    assigned_room_type = st.selectbox("Assigned Room Type", list("ABCDEFGH"))

with col6:
    booking_changes = st.number_input("Booking Changes",0,20,0)

st.divider()

# ====================================
# ADDITIONAL FEATURES
# ====================================

st.header("Additional Information")

col7, col8, col9 = st.columns(3)

with col7:
    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit","Non Refund","Refundable"]
    )

    agent = st.number_input("Agent ID",0,500,0)

with col8:
    company = st.number_input("Company ID",0,500,0)
    days_in_waiting_list = st.number_input("Days in Waiting List",0,400,0)

with col9:
    customer_type = st.selectbox(
        "Customer Type",
        ["Transient","Contract","Transient-Party","Group"]
    )

    adr = st.number_input("Average Daily Rate",0.0,1000.0,100.0)

st.divider()

col10, col11 = st.columns(2)

with col10:
    required_car_parking_spaces = st.number_input("Parking Spaces",0,10,0)

with col11:
    total_of_special_requests = st.number_input("Special Requests",0,10,0)

# ====================================
# CREATE INPUT DATA
# ====================================

input_data = pd.DataFrame({
    "hotel":[hotel],
    "lead_time":[lead_time],
    "arrival_date_year":[arrival_date_year],
    "arrival_date_month":[arrival_date_month],
    "arrival_date_week_number":[arrival_date_week_number],
    "arrival_date_day_of_month":[arrival_date_day_of_month],
    "stays_in_weekend_nights":[stays_in_weekend_nights],
    "stays_in_week_nights":[stays_in_week_nights],
    "adults":[adults],
    "children":[children],
    "babies":[babies],
    "meal":[meal],
    "country":[country],
    "market_segment":[market_segment],
    "distribution_channel":[distribution_channel],
    "is_repeated_guest":[is_repeated_guest],
    "previous_cancellations":[previous_cancellations],
    "previous_bookings_not_canceled":[previous_bookings_not_canceled],
    "reserved_room_type":[reserved_room_type],
    "assigned_room_type":[assigned_room_type],
    "booking_changes":[booking_changes],
    "deposit_type":[deposit_type],
    "agent":[agent],
    "company":[company],
    "days_in_waiting_list":[days_in_waiting_list],
    "customer_type":[customer_type],
    "adr":[adr],
    "required_car_parking_spaces":[required_car_parking_spaces],
    "total_of_special_requests":[total_of_special_requests]
})

# ====================================
# FEATURE ENGINEERING
# ====================================

input_data["total_guest"] = (
    input_data["adults"] +
    input_data["children"] +
    input_data["babies"]
)

input_data["total_nights"] = (
    input_data["stays_in_weekend_nights"] +
    input_data["stays_in_week_nights"]
)

input_data["booking_type"] = input_data["hotel"]

# ====================================
# PREDICTION
# ====================================

if st.button("Predict Cancellation"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Booking will likely be **CANCELED**")
    else:
        st.success("✅ Booking will likely **NOT be canceled**")

    st.metric(
        label="Cancellation Probability",
        value=f"{probability:.2%}"
    )

st.divider()

st.caption("Model: LightGBM Tuned | Streamlit App | Project 2026")