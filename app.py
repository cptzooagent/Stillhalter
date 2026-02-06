import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Dashboard v2", layout="wide")

# --- 1. MATHE-KERN: DELTA-BERECHNUNG (BLACK-SCHOLES) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet das theoretische Delta für die Risiko-Einschätzung."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN (YAHOO FINANCE - 15M DELAY) ---
@st.cache_data(ttl=900) # 15 Minuten Cache
def get_yf_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        return tk, price, tk.options
    except:
        return None, None, []

# --- UI START ---
st.title("🛡️ CapTrader Pro Stillhalter Dashboard")
st.info("🚀 System läuft jetzt auf Yahoo Finance Basis (15 Min. Verzögerung).")

# --- SEKTION 1: MARKT-SCANNER ---
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta 0.15)")
if st.button("🚀 Markt-Scan jetzt starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    results = []
    
    with st.spinner("Scanne Märkte & berechne Deltas..."):
        for t in watchlist:
            tk, price, dates = get_yf_data(t)
            if dates and price:
                # Suche Laufzeit nah an 30 Tagen
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta für die Chain schätzen
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                chain['diff'] = (chain['delta'].abs() - 0.15).abs()
                best = chain.sort_values('diff').iloc[0]
                
                days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                y_pa = (best['bid'] / best['strike']) * (365 / max(1, days)) * 100
                results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'days': days})

    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                with st.container(border=True):
                    st.write(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%")
                    st.caption(f"Strike: {row['strike']:.1f}$ | {row['days']} T.")

st.divider()

# --- SEKTION 2: DEPOT & REPAIR-AMPEL ---
st.subheader("💼 Mein Depot & Repair-Status")

# Deine Daten aus dem Screenshot
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
    _, curr, _ = get_yf_data(item['Ticker'])
    if curr:
        diff = (curr / item['Einstand'] - 1) * 100
        # Ampel Logik
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {curr:.2f}$ ({diff:.1f}%)")

st.divider()

# --- SEKTION 3: PRÄZISIONS-FINDER ---
st.subheader("🔍 Options-Finder (15m Delay)")
c1, c2 = st.columns([1, 2])
with c1:
    option_mode = st.radio("Strategie", ["put", "call"], horizontal=True)
with c2:
    search_ticker = st.text_input("Ticker-Symbol", value="HOOD").upper()

if search_ticker:
    tk, price, dates = get_yf_data(search_ticker)
    if tk and price:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        chosen_date = st.selectbox("Laufzeit wählen", dates)
        
        chain = tk.option_chain(chosen_date).puts if option_mode == "put" else tk.option_chain(chosen_date).calls
        T = (datetime.strptime(chosen_date, '%Y-%m-%d') - datetime.now()).days / 365
        
        # OTM Filter
        if option_mode == "put":
            df = chain[chain['strike'] < price].sort_values('strike', ascending=False)
        else:
            df = chain[chain['strike'] > price].sort_values('strike', ascending=True)
            
        for _, opt in df.head(6).iterrows():
            iv = opt['impliedVolatility'] or 0.4
            calc_delta = calculate_bsm_delta(price, opt['strike'], T, iv, option_type=option_mode)
            puffer = (abs(opt['strike'] - price) / price) * 100
            
            risk_color = "🟢" if abs(calc_delta) < 0.16 else "🟡" if abs(calc_delta) < 0.31 else "🔴"
            
            with st.expander(f"{risk_color} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                st.write(f"**Delta (BSM):** {abs(calc_delta):.2f}")
                st.write(f"**Sicherheitspuffer:** {puffer:.1f}%")
                st.write(f"**Implizite Vola:** {iv*100:.1f}%")
