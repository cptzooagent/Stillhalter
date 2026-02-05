import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Scanner Mittelweg", layout="centered")

# --- DER TRICK: DATEN AUF DISK SPEICHERN ---
# ttl="1d" bedeutet: Die Daten werden nur 1x am Tag wirklich neu geladen.
@st.cache_data(persist="disk", ttl="1d")
def get_persistent_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    options = stock.options
    # Wir geben nur die wichtigsten Infos zurück, um Speicher zu sparen
    return {"hist": hist, "options": options, "price": hist['Close'].iloc[-1]}

st.title("🛡️ Stillhalter Scanner")

# Ticker Auswahl
ticker_symbol = st.text_input("Aktien-Ticker", "MSFT").upper()

if ticker_symbol:
    try:
        # Daten laden (entweder frisch oder aus dem Speicher)
        data = get_persistent_data(ticker_symbol)
        
        st.metric("Kurs (gespeichert)", f"{data['price']:.2f} $")
        st.line_chart(data['hist']['Close'].tail(60))

        if data['options']:
            expiry = st.selectbox("Ablaufdatum", data['options'])
            
            # Optionsketten sind leider schwer zu persistieren, 
            # daher laden wir sie hier "vorsichtig" live.
            stock = yf.Ticker(ticker_symbol)
            puts = stock.option_chain(expiry).puts
            
            st.subheader(f"Puts für {expiry}")
            st.write(puts[['strike', 'lastPrice', 'volatility']].head(5))

    except Exception as e:
        st.error(f"Daten gerade nicht verfügbar: {e}")

# Button zum manuellen Löschen des Speichers
if st.button("Daten erzwingen neu laden"):
    st.cache_data.clear()
    st.rerun()
