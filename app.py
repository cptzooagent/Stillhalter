import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- BASIS-KONFIGURATION ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. FINANZ-MATHE-KERN ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    except: return 0

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).iloc[-1]

# --- 2. DATEN-ABRUF-LOGIK ---
@st.cache_data(ttl=900)
def get_full_market_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        info = tk.fast_info
        price = info['last_price']
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']) if not hist.empty else 50
        
        earn_date = "N/A"
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        return price, list(tk.options), earn_date, rsi_val
    except: return None, [], "N/A", 50

# --- 3. HAUPT-UI ---
st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION A: DEPOT-WERTE ---
st.subheader("📊 Meine Depot-Performance")
d_col1, d_col2, d_col3 = st.columns(3)
d_col1.metric("Depotwert", "42.500 $", "+1.2%")
d_col2.metric("Verfügbare Margin", "28.400 $", "-0.5%")
d_col3.metric("Monatlicher Cashflow", "1.150 $", "8%")

# --- SEKTION B: MARKT-SCAN ---
st.write("---")
if st.button("🚀 Markt-Scan (Top 12 Watchlist)"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "COIN"]
    cols = st.columns(3)
    for i, t in enumerate(watchlist):
        p, dts, earn, rsi = get_full_market_data(t)
        if p:
            with cols[i % 3]:
                st.info(f"**{t}** | Kurs: {p:.2f}$")
                st.caption(f"RSI: {rsi:.0f} | Earnings: {earn}")

# --- SEKTION C: EINZEL-CHECK (Der gefixte Bereich) ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")

c1, c2 = st.columns([1, 3])
strat = c1.radio("Typ", ["put", "call"], horizontal=True)
ticker = c2.text_input("Symbol eingeben", "HOOD").upper()

if ticker:
    price, dates, earn, rsi = get_full_market_data(ticker)
    if price and dates:
        st.write(f"Aktueller Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        expiry = st.selectbox("Laufzeit wählen", dates)
        
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry).calls if strat == "call" else tk.option_chain(expiry).puts
        T = max(1/365, (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365)
        
        # Logik für korrekte OTM-Strikes (Fix für Bild 27)
        if strat == "put":
            filtered_chain = chain[chain['strike'] < price].sort_values('strike', ascending=False)
        else:
            filtered_chain = chain[chain['strike'] > price].sort_values('strike', ascending=True)
        
        for _, opt in filtered_chain.head(8).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=strat)
            d_abs = abs(delta)
            
            # Farb-Logik
            if d_abs < 0.16: color, label = "🟢", "(Sicher)"
            elif d_abs < 0.31: color, label = "🟡", "(Moderat)"
            else: color, label = "🔴", "(Aggressiv)"
            
            with st.expander(f"{color} {label} Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f}"):
                st.write(f"💰 Prämie: {opt['bid']*100:.0f}$ | OTM-Chance: {int((1-d_abs)*100)}%")
