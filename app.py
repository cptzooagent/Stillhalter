import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- KONFIGURATION ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. MATHE-FUNKTIONEN (SICHER GEKAPSELT) ---
def get_delta(S, K, T, sigma, opt_type='put'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0
    try:
        d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    except: return 0

def get_rsi(ticker_obj):
    try:
        df = ticker_obj.history(period="1mo")
        if df.empty: return 50
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).iloc[-1]
    except: return 50

# --- 2. DATEN-ABRUF ---
@st.cache_data(ttl=600)
def fetch_basic_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        return price, list(tk.options), tk
    except: return None, [], None

# --- UI START ---
st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: MEINE DEPOT-WERTE (FIX) ---
st.subheader("💼 Meine Depot-Performance")
d1, d2, d3 = st.columns(3)
# Hier können Sie Ihre echten Werte eintragen
d1.metric("Depotwert", "42.500 $", "+1.2%")
d2.metric("Verfügbare Margin", "28.400 $", "-0.5%")
d3.metric("Cashflow (MTD)", "1.150 $", "8%")

# --- SEKTION 2: MARKT-SCAN ---
st.write("---")
if st.button("🚀 Markt-Scan (Watchlist)"):
    watchlist = ["NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "COIN", "AAPL", "MSFT"]
    cols = st.columns(3)
    for i, t in enumerate(watchlist):
        p, _, _ = fetch_basic_data(t)
        if p:
            with cols[i % 3]:
                st.info(f"**{t}** | Kurs: {p:.2f}$")

# --- SEKTION 3: EINZEL-CHECK (STRIKE-FIX FÜR BILD 27) ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")

check_col1, check_col2 = st.columns([1, 3])
s_type = check_col1.radio("Strategie", ["put", "call"], horizontal=True)
s_ticker = check_col2.text_input("Symbol prüfen", "HOOD").upper()

if s_ticker:
    price, dates, tk_obj = fetch_basic_data(s_ticker)
    if price and dates:
        rsi = get_rsi(tk_obj)
        st.write(f"Aktueller Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        expiry = st.selectbox("Laufzeit wählen", dates)
        
        # Optionskette holen
        chain = tk_obj.option_chain(expiry).calls if s_type == "call" else tk_obj.option_chain(expiry).puts
        days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
        T = max(1/365, days / 365)
        
        # WICHTIGSTER FIX: Nur Strikes anzeigen, die OTM (Out of the Money) sind!
        # Das verhindert die 200$ Strikes bei 80$ Kurs (Bild 27)
        if s_type == "put":
            filtered = chain[chain['strike'] < price].sort_values('strike', ascending=False)
        else:
            filtered = chain[chain['strike'] > price].sort_values('strike', ascending=True)
            
        if filtered.empty:
            st.warning("Keine passenden Strikes gefunden.")
        else:
            for _, opt in filtered.head(8).iterrows():
                delta = abs(get_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, s_type))
                
                # Risiko-Ampel
                color = "🟢" if delta < 0.15 else "🟡" if delta < 0.25 else "🔴"
                
                with st.expander(f"{color} Strike {opt['strike']:.1f}$ | Delta: {delta:.2f}"):
                    c1, c2 = st.columns(2)
                    c1.write(f"💰 **Prämie:** {opt['bid']*100:.0f}$")
                    c1.write(f"📉 **Abstand:** {abs(opt['strike']-price)/price*100:.1f}%")
                    c2.write(f"🎯 **OTM-Chance:** {int((1-delta)*100)}%")
                    c2.write(f"🌊 **IV:** {int((opt['impliedVolatility'] or 0)*100)}%")
