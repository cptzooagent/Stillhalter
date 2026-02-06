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

# --- 2. DATEN-FUNKTIONEN (ROBUST VERSION) ---
@st.cache_data(ttl=86400)
def get_auto_watchlist():
    high_yield_base = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR", "AI", "SNOW", "SHOP", "PYPL", "ABNB"]
    return high_yield_base

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        inf = tk.fast_info
        price = inf['last_price']
        dates = list(tk.options)
        
        # Sicherer Earnings-Check
        has_e, e_dt = False, None
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                # Wir nehmen das erste Datum aus der Liste/Serie
                dt_raw = cal['Earnings Date']
                if isinstance(dt_raw, (list, pd.Series)) and len(dt_raw) > 0:
                    next_e = dt_raw[0].replace(tzinfo=None)
                    days_to = (next_e - datetime.now()).days
                    if 0 <= days_to <= 14:
                        has_e, e_dt = True, next_e.strftime('%d.%m.')
        except: 
            pass # Earnings-Fehler ignorieren, damit Preis trotzdem geladen wird
            
        return price, dates, has_e, e_dt
    except:
        return None, [], False, None

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Gewünschte Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

st.title("🛡️ CapTrader AI Market Scanner")

# --- 3. SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = get_auto_watchlist()
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, has_e, e_dt = get_stock_basics(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    best = safe_opts.sort_values('delta_val', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    y_pa = (best['bid'] / best['strike']) * (365 / days) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'t': t, 'y': y_pa, 's': best['strike'], 'b': best['bid'], 'p': (abs(best['strike']-price)/price)*100, 'he': has_e, 'ed': e_dt})
            except: continue

    if results:
        opp_df = pd.DataFrame(results).sort_values('p', ascending=False).head(10)
        cols = st.columns(5)
        for idx, (_, r) in enumerate(opp_df.iterrows()):
            with cols[idx % 5]:
                st.markdown(f"### {r['t']}")
                if r['he']: st.warning(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"💰 Bid: **{r['b']:.2f}$** | Yield: **{r['y']:.1f}%**")
                st.write(f"🎯 Strike: **{r['s']:.1f}$**")

st.write("---")

# --- 4. DEPOT ---
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "GTM", "HIMS", "HOOD", "JKS", "NVO", "RBRK", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot_list):
    price, _, has_e, e_dt = get_stock_basics(t)
    if price:
        with d_cols[i % 4]:
            st.write(f"**{t}**: {price:.2f}$" + (f" ⚠️ ({e_dt})" if has_e else ""))

st.write("---")

# --- 5. EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()

if t_in:
    price, dates, has_e, e_dt = get_stock_basics(t_in)
    if price and dates:
        if has_e: st.error(f"⚠️ Earnings am {e_dt}!")
        st.write(f"Kurs: **{price:.2f}$**")
        d_sel = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < price].sort_values('strike', ascending=False).head(6) if mode == "put" else chain[chain['strike'] > price].sort_values('strike', ascending=True).head(6)
        
        for _, opt in df.iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, mode)
            with st.expander(f"Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                st.write(f"💰 **Einnahme: {opt['bid']*100:.0f}$** | OTM: {(1-abs(delta))*100:.1f}%")
                st.write(f"Puffer: {(abs(opt['strike']-price)/price)*100:.1f}% | Delta: {abs(delta):.2f}")
