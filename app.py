import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

st.set_page_config(page_title="Pro-Scanner", layout="centered")

# API Key aus den Secrets laden
API_KEY = st.secrets["ALPHA_VANTAGE_KEY"]

@st.cache_data(ttl=3600)
def get_price_alpha(symbol):
    # Alpha Vantage als stabile Datenquelle
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}'
    r = requests.get(url)
    data = r.json()
    return float(data['Global Quote']['05. price'])

st.title("🛡️ Pro-Stillhalter")

ticker_symbol = st.text_input("Ticker (z.B. MSFT)", "MSFT").upper()

if ticker_symbol:
    try:
        # 1. Kurs über Alpha Vantage (Stabil)
        current_price = get_price_alpha(ticker_symbol)
        st.metric("Echtzeit-Kurs", f"{current_price:.2f} $")
        
        # 2. Optionen über yfinance (Backup/Live)
        stock = yf.Ticker(ticker_symbol)
        expirations = stock.options
        
        if expirations:
            expiry = st.selectbox("Ablaufdatum", expirations)
            dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            
            # Chain laden
            puts = stock.option_chain(expiry).puts
            filtered = puts[puts['strike'] <= current_price * 0.95].sort_values('strike', ascending=False)
            
            st.subheader(f"Premium-Optionen ({max(1, dte)} Tage)")
            for _, row in filtered.head(5).iterrows():
                ann_return = (row['lastPrice'] / row['strike']) * (365 / max(1, dte)) * 100
                with st.expander(f"Strike {row['strike']}$ | {ann_return:.1f}% p.a."):
                    st.write(f"Prämie: **{row['lastPrice']}$**")
                    st.write(f"Sicherheitspuffer: {((current_price/row['strike'])-1)*100:.1f}%")
        
    except Exception as e:
        st.warning("Warte auf Daten-Update... (Alpha Vantage Limits beachten)")
        st.info("Hinweis: Im kostenlosen Modus sind 5 Abfragen pro Minute erlaubt.")
