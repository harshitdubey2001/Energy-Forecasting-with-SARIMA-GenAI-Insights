import streamlit as st
import pandas as pd
import json
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from src.preprocessing import load_energy_data, clean_and_resample
from src.genai_layer import generate_forecast_insights, answer_user_query
from src.utils import plot_forecast

# --------------------------------------------------------
# 1. Load saved model + metadata
# --------------------------------------------------------

MODEL_PATH = "../models/sarima_model.pkl"
META_PATH = "../models/metadata.json"

@st.cache_resource
def load_model():
    model = SARIMAXResults.load(MODEL_PATH)
    return model

@st.cache_resource
def load_metadata():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return meta


model_fit = load_model()
meta = load_metadata()


# --------------------------------------------------------
# 2. App Layout
# --------------------------------------------------------

st.title("⚡ Energy Load Forecasting with SARIMA + GenAI")
st.write(f"Model last trained on: **{meta['last_trained']}**")
st.write(f"Model order: `{meta['order']}`")
st.write(f"Seasonal order: `{meta['seasonal_order']}`")


# --------------------------------------------------------
# 3. Load and process dataset
# --------------------------------------------------------

df = load_energy_data("../data/PJME_hourly.csv")
daily_df = clean_and_resample(df)

st.subheader("📉 Historical Daily Energy Load")
st.line_chart(daily_df)


# --------------------------------------------------------
# 4. Generate Forecast
# --------------------------------------------------------

steps = st.slider("Forecast Days", 7, 60, 30)

if st.button("Generate Forecast"):
    forecast = model_fit.forecast(steps=steps)
    st.subheader("🔮 Forecast Output")
    st.write(forecast)

    st.subheader("📈 Forecast Chart")
    st.line_chart(forecast)

    # Store forecast for GenAI section
    st.session_state["forecast_values"] = forecast.to_list()


# --------------------------------------------------------
# 5. GenAI Insights Section
# --------------------------------------------------------

st.subheader("🤖 AI-Generated Forecast Insights")

api_key = st.text_input("Enter GROK API Key:", type="password")

if st.button("Generate Insights"):
    if "forecast_values" not in st.session_state:
        st.error("Please generate forecast first.")
    else:
        values = st.session_state["forecast_values"]
        insights = generate_forecast_insights(values)
        st.write(insights)


# --------------------------------------------------------
# 6. Natural Language Q&A Section
# --------------------------------------------------------

st.subheader("💬 Ask Questions About The Forecast")

user_query = st.text_input("Ask something like: Why is demand higher around day 10?")

if st.button("Ask AI"):
    if not api_key:
        st.error("Enter your API key first.")
    elif "forecast_values" not in st.session_state:
        st.error("Generate forecast first.")
    else:
        values = st.session_state["forecast_values"]
        response = answer_user_query(api_key, user_query, values)
        st.write(response)
