import streamlit as st
from PIL import Image

st.title("📊 Analytics")

st.image(
    "reports/target_distribution.png",
    caption="Target Distribution"
)

st.image(
    "reports/correlation_matrix.png",
    caption="Correlation Matrix"
)