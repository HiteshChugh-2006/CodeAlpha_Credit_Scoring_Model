import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Scoring System",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "models/best_credit_model.pkl"
    )

model = load_model()

st.title("🏦 Credit Scoring Prediction System")

st.write(
    "Predict whether a customer is creditworthy."
)

st.sidebar.header("Customer Details")

duration = st.sidebar.number_input(
    "Duration (Months)",
    1,
    72,
    12
)

credit_amount = st.sidebar.number_input(
    "Credit Amount",
    100,
    50000,
    2000
)

age = st.sidebar.number_input(
    "Age",
    18,
    100,
    30
)

checking_status = st.sidebar.selectbox(
    "Checking Status",
    [
        "<0",
        "0<=X<200",
        ">=200",
        "no checking"
    ]
)

input_data = pd.DataFrame(
    {
        "checking_status":[checking_status],
        "duration":[duration],
        "credit_history":["existing paid"],
        "purpose":["radio/tv"],
        "credit_amount":[credit_amount],
        "savings_status":["<100"],
        "employment":["1<=X<4"],
        "installment_commitment":[2],
        "personal_status":["male single"],
        "other_parties":["none"],
        "residence_since":[2],
        "property_magnitude":["car"],
        "age":[age],
        "other_payment_plans":["none"],
        "housing":["own"],
        "existing_credits":[1],
        "job":["skilled"],
        "num_dependents":[1],
        "own_telephone":["yes"],
        "foreign_worker":["yes"],
        "amount_per_month":[
            credit_amount/(duration+1)
        ]
    }
)

if st.button("Predict Credit Risk"):

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success(
            "Approved Customer"
        )
    else:
        st.error(
            "High Risk Customer"
        )

    st.metric(
        "Risk Probability",
        f"{probability:.2%}"
    )