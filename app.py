import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

# --- 1. MATHE: DELTA ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call': return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    base = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
        df = pd.read_csv(url, header=None, names=['T'])
        return list(set(base + df['T'].head(80).tolist()))
    except: return base

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn, ed = False, None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            edt = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            if 0 <= (edt - datetime.now()).days <= 14:
                earn, ed = True, edt.strftime('%d.%m.')
        return {"price": price, "dates": dates, "earn": earn, "ed": ed}
    except: return None

# --- UI SETUP ---
st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")
st.sidebar.header("🛡️ Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_y = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = random.sample(get_auto_watchlist(), 50)
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
                chain['dv'] = chain.apply(lambda r: calculate_bsm_delta(data['price'], r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe = chain[chain['dv'].abs() <= max_delta].copy()
                if not safe.empty:
                    best = safe.sort_values('dv', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    yield_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    if yield_pa >= min_y:
                        results.append({'t': t, 'y': yield_pa, 's': best['strike'], 'b': best['bid'], 'p': (abs(best['strike'] - data['price']) / data['price']) * 100, 'earn': data['earn'], 'ed': data['ed']})
            except: continue
    if results:
        opp_df = pd.DataFrame(results).sort_values('p', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, r) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {r['t']}")
                if r['earn']: st.caption(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"💰 Bid: **{r['b']:.2f}$** | Yield: **{r['y']:.1f}%**")
                st.write(f"🎯 Strike: **{r['s']:.1f}$**")

st.write("---")

# --- SEKTION 2: DEPOT ---
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot_list):
    data = get_stock_data(t)
    if data:
        with d_cols[i % 4]: st.write(f"**{t}**: {data['price']:.2f}$")

st.write("---")

# --- SEKTION 3: EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()
if t_in:
    data = get_stock_data(t_in)
    if data and data['dates']:
        if data['earn']: st.warning(f"⚠️ Earnings am {data['ed']}!")
        d_sel = st.selectbox("Laufzeit", data['dates'])
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < data['price']].sort_values('strike', ascending=False) if mode == "put" else chain[chain['strike'] > data['price']].sort_values('strike', ascending=True)
        for _, opt in df.head(6).iterrows():
            delta = calculate_bsm_delta(data['price'], opt['strike'], T, opt['impliedVolatility'] or 0.4, mode)
            with st.expander(f"Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                st.write(f"💰 **Einnahme: {opt['bid']*100:.0f}$** | OTM: **{(1-abs(delta))*100:.1f}%**")
                st.write(f"🎯 Puffer: {(abs(opt['strike']-data['price'])/data['price'])*100:.1f}% | Delta: {abs(delta):.2f}")
