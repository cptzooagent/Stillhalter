import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from engine import calculate_bsm_delta, calculate_rsi, get_clean_earnings

# --- CONFIG ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- DATA FETCHING (mit Caching) ---
@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_date = get_clean_earnings(tk)
        earn_str = earn_date.strftime('%d.%m.') if earn_date else ""
        return price, dates, earn_str, rsi_val, earn_date
    except:
        return None, [], "", 50, None

# --- UI SECTIONS ---
st.title("🛡️ CapTrader AI - Modular Scanner")

# Hier nutzen wir eine einfache Sidebar für globale Filter
st.sidebar.header("Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100

# Beispiel: Einzel-Check mit der neuen Engine
t_in = st.text_input("Ticker Symbol", value="NVDA").upper()

if t_in:
    price, dates, earn, rsi, earn_dt = get_stock_data(t_in)
    if price:
        st.success(f"{t_in} Kurs: {price:.2f}$ | RSI: {rsi:.0f}")
        
        if dates:
            d_sel = st.selectbox("Laufzeit", dates)
            # Hier würde die Logik für die Option Chain folgen (wie im vorigen Code)
            # ...
