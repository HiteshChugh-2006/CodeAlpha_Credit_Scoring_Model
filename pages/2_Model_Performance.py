import streamlit as st

st.title("📈 Model Performance")

st.image(
    "reports/confusion_matrix.png"
)

try:
    with open(
        "reports/metrics.txt",
        "r"
    ) as f:

        st.code(f.read())

except:
    st.warning(
        "Metrics file not found"
    )