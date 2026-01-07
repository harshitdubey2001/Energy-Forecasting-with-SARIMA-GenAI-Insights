import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env file!")

def generate_forecast_insights(forecast_values):
    llm = ChatGroq(api_key=GROQ_KEY,model="llama-3.3-70b-versatile",temperature=0.2)

    prompt = f"""
    You are an expert energy analyst.

    The following are forecasted energy load values for the next 30 days:
    {forecast_values}

    Provide:
    1. Trend direction
    2. Seasonal behavior
    3. Peak demand days
    4. Risks or anomalies
    5. Business recommendations
    """

    response = llm.invoke(prompt)

    return response.content

def answer_user_query(api_key,query,forecast_values):
    llm = ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile",temperature=0.2)

    prompt = f"""
    You are an expert energy analyst.

    The following are forecasted energy load values for the next 30 days:
    {forecast_values}

    {query}
    """

    reponse = llm.invoke(prompt)

    return reponse.content
