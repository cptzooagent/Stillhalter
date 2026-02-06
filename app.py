import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    """Holt Ticker aus stabiler Quelle und ergänzt High-IV Favoriten."""
    high_yield_base = [
        "TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", 
        "UPST", "HOOD", "SOFI", "MSTR", "AI", "SNOW", "SHOP", "PYPL", "ABNB"
    ]
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
        response = pd.read_csv(url, header=None, names=['Ticker'])
        nasdaq_list = response['Ticker'].head(100).tolist()
        return list(set(high_yield_base + nasdaq_list))
    except:
        return high_yield_base

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        return price, dates
    except:
        return None, []

# --- UI: SEITENLEISTE (SICHERHEITS-FILTER) ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Gewünschte Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")
st.write(f"Suche nach Puts mit Delta ≤ **{max_delta:.2f}** (Sicherheits-Puffer-Fokus).")

# SEKTION 1: AUTOMATISCHER SCANNER
if st.button("🚀 Markt-Scan mit Sicherheits-Filter starten"):
    full_watchlist = get_auto_watchlist()
    scan_list = random.sample(full_watchlist, min(len(full_watchlist), 60)) 
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(scan_list):
        status_text.text(f"Analysiere {t}...")
        progress_bar.progress((i + 1) / len(scan_list))
        
        price, dates = get_stock_basics(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta für alle berechnen
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                
                # Nur Optionen unter dem Delta-Limit
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    best = safe_opts.sort_values('delta_val', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    puffer = (abs(best['strike'] - price) / price) * 100
                    
                    if y_pa >= min_yield_pa:
                        results.append({'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'bid': best['bid'], 'puffer': puffer, 'delta': abs(best['delta_val']), 'days': days})
            except: continue

    status_text.text("Scan abgeschlossen!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('puffer', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Puffer", f"{row['puffer']:.1f}%")
                st.write(f"Yield: **{row['yield']:.1f}% p.a.**")
                st.write(f"Strike: **{row['strike']:.1f}$**")
                st.caption(f"Delta: {row['delta']:.2f}")

st.write("---") # Ersatz für st.divider()

# SEKTION 2: DEPOT
st.subheader("💼 Depot-Status")
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

st.write("---") # Ersatz für st.divider()

# SEKTION 3: FINDER
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()
if t_in:
    price, dates = get_stock_basics(t_in)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$**")
        d_sel = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < price].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > price].sort_values('strike', ascending=True)
        for _, opt in df.head(5).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
            with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$"):
                st.write(f"Delta: {abs(delta):.2f} | Puffer: {(abs(opt['strike']-price)/price)*100:.1f}%")
