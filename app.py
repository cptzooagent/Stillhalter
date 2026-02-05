import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Scanner", layout="centered")

# 1. Kursdaten cachen (Funktioniert, da es ein einfacher DataFrame ist)
@st.cache_data(ttl=600)
def get_historical_data(symbol):
    t = yf.Ticker(symbol)
    return t.history(period="1y")

# 2. Options-Liste cachen
@st.cache_data(ttl=3600)
def get_expiry_dates(symbol):
    t = yf.Ticker(symbol)
    return t.options

st.title("Scanner")

ticker_symbol = st.text_input("Aktien-Ticker (z.B. AAPL)", "AAPL").upper()
max_delta_input = st.slider("Max. Delta (für Puts)", 0.05, 0.40, 0.20, step=0.01)

if ticker_symbol:
    try:
        hist = get_historical_data(ticker_symbol)
        
        if hist.empty:
            st.error("Keine Daten gefunden.")
        else:
            current_price = hist['Close'].iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            support_level = hist['Low'].tail(60).min()

            st.metric("Kurs", f"{current_price:.2f} $")
            
            # Trend-Check
            if current_price > sma_200:
                st.success(f"Trend: Bullish (über SMA 200)")
            else:
                st.warning(f"Trend: Bearish (unter SMA 200)")

            st.write(f"Support (60T): **{support_level:.2f} $**")
            st.line_chart(hist['Close'].tail(90))

            # Optionen laden
            expirations = get_expiry_dates(ticker_symbol)
            if expirations:
                expiry = st.selectbox("Ablaufdatum", expirations)
                dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                
                # Dieser Teil bleibt live
                t_obj = yf.Ticker(ticker_symbol)
                puts = t_obj.option_chain(expiry).puts
                
                # Filter & Berechnung
                filtered = puts[puts['strike'] <= current_price].sort_values('strike', ascending=False)
                
                st.subheader(f"Puts ({max(1, dte)} Tage)")
                for _, row in filtered.head(5).iterrows():
                    ann_return = (row['lastPrice'] / row['strike']) * (365 / max(1, dte)) * 100
                    with st.expander(f"Strike {row['strike']}$ | {ann_return:.1f}% p.a."):
                        st.write(f"Prämie: **{row['lastPrice']}$**")
                        st.write(f"Abstand: {((current_price/row['strike'])-1)*100:.1f}%")
            else:
                st.info("Keine Optionen verfügbar.")

    except Exception as e:
        st.error(f"Yahoo-Limit erreicht oder Fehler: {e}")
        st.info("Tipp: Warte kurz oder versuche einen sehr bekannten Ticker wie MSFT.")
