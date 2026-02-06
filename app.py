import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Dashboard", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    """Holt Kurs und verfügbare Daten ohne komplexe Objekte zu cachen."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        return price, dates
    except:
        return None, []

# --- UI START ---
st.title("🛡️ CapTrader Pro Dashboard")
st.caption("Daten: Yahoo Finance (15m Delay) | Delta: BSM Berechnung")

# SEKTION 1: SCANNER
st.subheader("💎 Top Gelegenheiten (Delta 0.15)")
if st.button("🚀 Markt-Scan jetzt starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    results = []
    
    with st.spinner("Scanne Märkte & Prämien..."):
        for t in watchlist:
            price, dates = get_stock_basics(t)
            if price and dates:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta & Rendite p.a. berechnen
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                chain['diff'] = (chain['delta'].abs() - 0.15).abs()
                best = chain.sort_values('diff').iloc[0]
                
                days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                # Rendite p.a. basierend auf BID
                y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'bid': best['bid'], 'days': days})

    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container():
                    st.markdown(f"### {row['ticker']}")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.write(f"**Strike: {row['strike']:.1f}$**")
                    st.write(f"Prämie (Bid): **{row['bid']:.2f}$**")
                    st.caption(f"Laufzeit: {row['days']} Tage")

st.write("---")

# SEKTION 2: DEPOT
st.subheader("💼 Depot & Repair-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

p_cols = st.columns(4)
for i, item in enumerate(depot_data):
    price, _ = get_stock_basics(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")

st.write("---")

# SEKTION 3: FINDER (JETZT MIT PRÄMIEN)
st.subheader("🔍 Options-Finder")
c1, c2 = st.columns([1, 2])
with c1:
    mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2:
    ticker = st.text_input("Ticker", value="HOOD").upper()

if ticker:
    price, dates = get_stock_basics(ticker)
    if price and dates:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        date = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(date).puts if mode == "put" else tk.option_chain(date).calls
        T = (datetime.strptime(date, '%Y-%m-%d') - datetime.now()).days / 365
        
        # Filter OTM & Berechnung
        df = chain[chain['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > price].sort_values('strike', ascending=True)
        
        for _, opt in df.head(8).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
            
            with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                col1, col2 = st.columns(2)
                col1.write(f"**Delta:** {abs(delta):.2f}")
                col1.write(f"**Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                col2.write(f"**Bid:** {opt['bid']:.2f}$")
                col2.write(f"**Ask:** {opt['ask']:.2f}$")
                st.caption(f"Implizite Vola: {opt['impliedVolatility']*100:.1f}%")
