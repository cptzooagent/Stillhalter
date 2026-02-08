import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. MATHE: DELTA & RSI ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        return price, list(tk.options), earn_str, rsi_val
    except:
        return None, [], "", 50

# --- UI: SIDEBAR ---
st.sidebar.header("Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 99, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: MARKT-SCAN (Wie Bild 15) ---
if st.button("🚀 Markt-Scan starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "NFLX", "COIN"]
    results = []
    
    for t in watchlist:
        price, dates, earn, rsi = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max(1/365, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365)
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4, option_type='put'), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (1/T) * 100
                    best = safe_opts.sort_values('y_pa', ascending=False).iloc[0]
