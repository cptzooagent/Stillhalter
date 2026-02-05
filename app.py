import streamlit as st
import yfinance as ticker_data
import pandas as pd
from datetime import datetime

# App-Konfiguration für Mobile
st.set_page_config(page_title="Stillhalter Scanner", layout="centered")

st.title("🛡️ Stillhalter Tool")

# 1. Eingabe & Einstellungen
ticker_symbol = st.text_input("Aktien-Ticker (z.B. AAPL, TSLA)", "AAPL").upper()
max_delta = st.slider("Max. Delta (für Puts)", 0.05, 0.40, 0.20, step=0.01)

if ticker_symbol:
    stock = ticker_data.Ticker(ticker_symbol)
    
    # 2. Trend & Support (SMA 200)
    hist = stock.history(period="1y")
    current_price = hist['Close'].iloc[-1]
    sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
    support_level = hist['Low'].tail(30).min() # Tief der letzten 30 Tage
    
    st.metric("Kurs", f"{current_price:.2f} $", f"{((current_price/sma_200)-1)*100:.1f}% über SMA200")

    # 3. Options-Chain laden
    expirations = stock.options
    if expirations:
        expiry = st.selectbox("Ablaufdatum wählen", expirations)
        opts = stock.option_chain(expiry)
        puts = opts.puts

        # 4. Filter & Berechnung
        # Wir berechnen den Annualized Return für alle Puts
        today = datetime.now()
        expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
        dte = (expiry_dt - today).days
        if dte <= 0: dte = 1 # Fehlervermeidung

        # Filter: Nur Puts unterhalb des gewählten Deltas (Näherung via Strike)
        # Hinweis: 'yfinance' liefert Greeks nicht immer stabil, oft nutzt man Strike vs. Kurs als Proxy
        filtered_puts = puts[puts['strike'] <= current_price * (1 - (max_delta * 0.5))] 
        
        # Berechnung der Rendite
        filtered_puts['Ann_Return'] = (filtered_puts['lastPrice'] / (filtered_puts['strike'] * 100)) * (365 / dte) * 100

        st.subheader(f"Gefilterte Puts (DTE: {dte})")
        
        # Anzeige der Ergebnisse
        for index, row in filtered_puts.sort_values(by='strike', ascending=False).head(5).iterrows():
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Strike: {row['strike']}$**")
                    st.write(f"Prämie: {row['lastPrice']}$")
                with col2:
                    st.write(f"**Return: {row['Ann_Return']:.1f}% p.a.**")
                    st.write(f"Volatilität: {row['impliedVolatility']:.1%}")
                st.divider()
    else:
        st.error("Keine Optionen gefunden.")