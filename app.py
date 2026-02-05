import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests

st.set_page_config(page_title="Pro-Scanner", layout="centered")

# API Key aus den Secrets laden
API_KEY = st.secrets["ALPHA_VANTAGE_KEY"]

@st.cache_data(ttl=600)
def get_price_alpha(symbol):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}'
    r = requests.get(url)
    data = r.json()
    return float(data['Global Quote']['05. price'])

st.title("🛡️ Pro-Stillhalter")

ticker_symbol = st.text_input("Ticker (z.B. MSFT)", "MSFT").upper()

if ticker_symbol:
    try:
        current_price = get_price_alpha(ticker_symbol)
        st.metric("Echtzeit-Kurs", f"{current_price:.2f} $")
        
        stock = yf.Ticker(ticker_symbol)
        expirations = stock.options
        
        if expirations:
            expiry = st.selectbox("Ablaufdatum", expirations)
            dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
            
            puts = stock.option_chain(expiry).puts
            # Nur Puts mit Volumen und unter dem aktuellen Preis
            filtered = puts[puts['strike'] <= current_price].sort_values('strike', ascending=False)
            
            selected_strike = st.selectbox("Strike für Profit-Profil wählen", filtered['strike'].head(10))
            
            # Prämie für den gewählten Strike finden
            premium = filtered[filtered['strike'] == selected_strike]['lastPrice'].values[0]
            
            # --- PROFIT PROFIL GRAFIK ---
            st.subheader("Gewinn/Verlust Profil")
            s_range = np.linspace(selected_strike * 0.8, current_price * 1.1, 100)
            # Gewinn = Prämie - Max(0, Strike - Kurs)
            profit = np.where(s_range >= selected_strike, premium * 100, (premium - (selected_strike - s_range)) * 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_range, y=profit, name='P&L at Expiry', line=dict(color='#00FF00' if profit[-1] > 0 else '#FF0000')))
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.add_vline(x=selected_strike, line_dash="dot", line_color="orange", annotation_text="Strike")
            fig.update_layout(title="GuV am Laufzeitende", xaxis_title="Aktienkurs", yaxis_title="Gewinn/Verlust $", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Break-Even: **{(selected_strike - premium):.2f} $**")

    except Exception as e:
        st.warning("Limit erreicht. Die Grafik lädt in Kürze neu...")
