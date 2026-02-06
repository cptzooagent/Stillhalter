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

# --- 2. DATEN-FUNKTIONEN (OPTIMIERT FÜR CACHING) ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    high_yield_base = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
        response = pd.read_csv(url, header=None, names=['Ticker'])
        return list(set(high_yield_base + response['Ticker'].head(100).tolist()))
    except:
        return high_yield_base

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    """Gibt nur serialisierbare Grunddaten zurück, um Cache-Fehler zu vermeiden."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        
        # Earnings Check
        has_earnings = False
        e_date = None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            earning_dt = tk.calendar['Earnings Date'][0]
            days_to = (earning_dt.replace(tzinfo=None) - datetime.now()).days
            if 0 <= days_to <= 14:
                has_earnings = True
                e_date = earning_dt.strftime('%d.%m.')
                
        return {"price": price, "dates": dates, "earnings": has_earnings, "e_date": e_date}
    except:
        return None

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Gewünschte Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")

# SEKTION 1: SCANNER
if st.button("🚀 Markt-Scan mit Sicherheits-Filter starten"):
    watchlist = random.sample(get_auto_watchlist(), 60)
    results = []
    progress = st.progress(0)
    
    for i, t in enumerate(watchlist):
        progress.progress((i + 1) / len(watchlist))
        data = get_stock_data(t)
        
        if data and data['dates']:
            try:
                tk = yf.Ticker(t)
                target_date = min(data['dates'], key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(data['price'], r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    best = safe_opts.sort_values('delta_val', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    
                    if y_pa >= min_yield_pa:
                        results.append({
                            't': t, 'y': y_pa, 's': best['strike'], 'b': best['bid'], 
                            'p': (abs(best['strike'] - data['price']) / data['price']) * 100,
                            'd': abs(best['delta_val']), 'earn': data['earnings'], 'ed': data['e_date']
                        })
            except: continue

    if results:
        opp_df = pd.DataFrame(results).sort_values('p', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, r) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {r['t']}")
                if r['earn']: st.caption(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"💰 Bid: **{r['b']:.2f}$**")
                st.write(f"📈 Yield: **{r['y']:.1f}%**")
                st.write(f"🎯 Strike: **{r['s']:.1f}$**")

st.write("---")

# SEKTION 2: DEPOT
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot_list):
    data = get_stock_data(t)
    if data:
        with d_cols[i % 4]:
            st.write(f"**{t}**: {data['price']:.2f}$")

st.write("---")

# SEKTION 3: EINZEL-CHECK
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()

if t_in:
    data = get_stock_data(t_in)
    if data and data['dates']:
        if data['earnings']: st.warning(f"⚠️ Earnings am {data['e_date']}!")
        st.write(f"Kurs: **{data['price']:.2f}$**")
        d_sel = st.selectbox("Laufzeit", data['dates'])
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < data['price']].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > data['price']].sort_values('strike', ascending=True)
        
        for _, opt in df.head(6).iterrows():
            delta = calculate_bsm_delta(data['price'], opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode)
            risk = "🟢" if abs(delta) < 0.16 else "🟡" if abs(delta) < 0.31 else "🔴"
            with st.expander(f"{risk} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                st.write(f"💰 **Einnahme pro Kontrakt: {opt['bid']*100:.0f}$**")
                st.write(f"📊 **Prob. OTM: {(1-abs(delta))*100:.1f}%** | Puffer: {(abs(opt['strike']-data['price'])/data['price'])*100:.1f}%")
