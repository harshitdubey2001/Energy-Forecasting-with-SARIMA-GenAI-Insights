# ⚡ Intelligent Energy Load Forecasting with SARIMA + GenAI (Groq Llama 3.3)

This project builds a complete **end-to-end energy load forecasting system** using:

- 📈 **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) for time-series forecasting  
- 🤖 **GenAI (Groq + Llama 3.3 70B)** for insight generation and natural language Q&A  
- 🖥️ **Streamlit** dashboard for interactive visualization  
- 🧩 Modular **Python architecture** for reusability and scalability  

The model predicts **daily energy demand** for the PJM East region (PJME) and explains the trends using an LLM.

---

## 📂 Project Structure

```
energy-forecasting/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── preprocessing.py
│   ├── model_sarima.py
│   ├── genai_layer.py
│   ├── utils.py
│
├── models/
│   ├── sarima_model.pickle
│   └── model_metadata.json
│
├── data/
│   └── energy_load.csv
│
├── notebooks/
│   └── EDA_and_SARIMA.ipynb
│
├── requirements.txt
└── README.md
```

Each folder and file is modular and designed for easy scaling and maintenance.

---

## 🚀 Features

### ✔ **Time-Series Forecasting with SARIMA**
- Weekly seasonality modeling
- Trend + autoregressive + moving average components
- Automatic hyperparameter search using Auto-ARIMA
- Model saved & reloaded for fast inference

---

### ✔ **Generative AI Insight Layer (Groq + Llama 3.3)**
Provides:
- Trend direction explanations  
- Peak demand days  
- Anomaly detection  
- Risk analysis  
- Business recommendations  
- Natural language Q&A about the forecast  

---

### ✔ **Streamlit Dashboard**
- Visualize historical data  
- Forecast next 7–60 days  
- Generate AI insights  
- Ask questions about energy demand patterns  

---

## 📊 Dataset

We use the **PJME Hourly Energy Load Dataset**, publicly available on Kaggle:

- Converted to daily averages  
- Missing values handled with time-based interpolation  
- Split into train/test for evaluation  

---

## 🧠 Model Used: SARIMA(1,1,3)(2,0,0,7)

Auto-ARIMA selected:

- **Order**: (1, 1, 3)  
- **Seasonal Order**: (2, 0, 0, 7)  

Which means:
- Weekly seasonality  
- Differencing for trend removal  
- Strong autoregressive and moving average structure  

---

## 🔮 Forecast Results

The model achieves:

- **MAE ≈ 2992**
- **RMSE ≈ 3765**
- About **7–9% error**, which is strong for energy forecasting

The forecast shows a smooth weekly cycle and realistic load variation.

---

## 🤖 GenAI Integration (Groq)

The project integrates **Groq Llama 3.3 70B**, using:

```
langchain-groq
groq
```

Add your API key in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

The LLM provides:
- Automated insights  
- Forecast explanation  
- Business suggestions  
- Natural-language Q&A  

---

## 🖥️ Running the Streamlit App

From the project root:

```
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser.

---

## 🛠 Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## 💾 Loading the Model

The SARIMA model is saved as:

```
models/sarima_model.pickle
models/model_metadata.json
```

Reload it with:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
model_fit = SARIMAXResults.load("models/sarima_model.pickle")
```

---

## 📐 Architecture Diagram

```
       ┌───────────────────────────────┐
       │        Data Source (PJME)     │
       └──────────────┬────────────────┘
                      ↓
       ┌───────────────────────────────┐
       │      Preprocessing Layer      │
       │  - resampling                 │
       │  - interpolation              │
       │  - train/test split           │
       └──────────────┬────────────────┘
                      ↓
       ┌───────────────────────────────┐
       │         SARIMA Model          │
       │  - auto_arima hyperparameter  │
       │  - weekly seasonality (m=7)   │
       │  - forecasting                │
       └──────────────┬────────────────┘
                      ↓
       ┌───────────────────────────────┐
       │        GenAI Layer (Groq)     │
       │  - trend insights             │
       │  - anomalies                  │
       │  - recommendations            │
       │  - user Q&A                   │
       └──────────────┬────────────────┘
                      ↓
       ┌───────────────────────────────┐
       │     Streamlit Dashboard        │
       └───────────────────────────────┘
```

---

## 🧩 Future Improvements

- ✓ Add Prophet model for comparison  
- ✓ Add TBATS for multi-seasonality  
- ✓ Add anomaly detection pipeline  
- ✓ Deploy on HuggingFace Spaces  
- ✓ Add caching for LLM calls  

---

## ⭐ Key Skills Demonstrated

- Time-series forecasting  
- SARIMA modeling  
- LLM integration  
- Streamlit development  
- Modular production-ready design  
- Data preprocessing & pipeline architecture  

---

## 📜 License

This project is open-source and free to use for learning or professional purposes.

