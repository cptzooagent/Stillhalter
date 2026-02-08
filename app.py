import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP ---
st.set_page_config(page_title="CapTrader Scanner", layout="wide")

# --- 2. FINANZ-MATHE ---
def get_delta(S, K, T, sigma, opt_type='put'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0
    d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1

# --- 3. DATEN-FUNKTION ---
@st.cache_data(ttl=600)
def get_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        return price, list(tk.options), tk
    except: return None, [], None

# --- 4. APP-STRUKTUR ---
st.title("🛡️ CapTrader AI Market Scanner")

# --- DEPOT (STATISCH FÜR STABILITÄT) ---
with st.container():
    st.subheader("💼 Mein Depot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Depotwert", "42.500 $", "+1.2%")
    c2.metric("Verfügbare Margin", "28.400 $", "-0.5%")
    c3.metric("Cashflow", "1.150 $", "8%")

# --- MARKT-SCAN ---
st.divider()
if st.button("🚀 Markt-Scan starten"):
    watchlist = ["NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "COIN"]
    cols = st.columns(len(watchlist))
    for i, t in enumerate(watchlist):
        p, _, _ = get_data(t)
        if p: cols[i].metric(t, f"{p:.2f}$")

# --- EINZEL-CHECK (STRIKE-FIX) ---
st.divider()
st.subheader("🔍 Einzel-Check")
col_a, col_b = st.columns([1, 2])
s_type = col_a.radio("Option", ["put", "call"])
s_ticker = col_b.text_input("Ticker", "HOOD").upper()

if s_ticker:
    p, dates, tk = get_data(s_ticker)
    if p and dates:
        exp = st.selectbox("Laufzeit", dates)
        chain = tk.option_chain(exp).calls if s_type == "call" else tk.option_chain(exp).puts
        T = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days / 365
        
        # Nur OTM Strikes!
        if s_type == "put":
            df = chain[chain['strike'] < p].sort_values('strike', ascending=False).head(5)
        else:
            df = chain[chain['strike'] > p].sort_values('strike', ascending=True).head(5)
            
        for _, row in df.iterrows():
            d = abs(get_delta(p, row['strike'], T, row['impliedVolatility'] or 0.4, s_type))
            with st.expander(f"Strike {row['strike']} | Delta {d:.2f}"):
                st.write(f"Prämie: {row['bid']*100:.0f}$ | Puffer: {abs(row['strike']-p)/p*100:.1f}%")
