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
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. DATEN-ENGINE (INKL. EARNINGS) ---
@st.cache_data(ttl=900)
def get_full_stock_info(symbol):
    """Holt Preis, Optionen UND Earnings-Status in einem Rutsch."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        
        # NEU: Earnings Check integriert
        has_earnings = False
        e_date = None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            next_e = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            days_to = (next_e - datetime.now()).days
            if 0 <= days_to <= 14:
                has_earnings = True
                e_date = next_e.strftime('%d.%m.')
        
        return {"price": price, "dates": dates, "has_e": has_earnings, "e_date": e_date}
    except:
        return None

# --- 3. UI SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

st.title("🛡️ CapTrader AI Market Scanner")

# --- 4. SEKTION: SCANNER ---
if st.button("🚀 Markt-Scan mit Earnings-Check starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "HOOD", "SOFI", "MSTR"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        data = get_full_stock_info(t)
        
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
                            'ticker': t, 'yield': y_pa, 'strike': best['strike'], 'bid': best['bid'], 
                            'puffer': (abs(best['strike'] - data['price']) / data['price']) * 100,
                            'earn': data['has_e'], 'e_date': data['e_date']
                        })
            except: continue

    if results:
        opp_df = pd.DataFrame(results).sort_values('puffer', ascending=False)
        cols = st.columns(4)
        for idx, (_, row) in enumerate(opp_df.iterrows()):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                if row['earn']: st.warning(f"⚠️ Earnings: {row['e_date']}")
                st.metric("Puffer", f"{row['puffer']:.1f}%")
                st.write(f"💰 Prämie: **{row['bid']:.2f}$** ({row['bid']*100:.0f}$)") 
                st.write(f"📈 Yield: **{row['yield']:.1f}% p.a.**")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$**")

st.write("---") 

# --- 5. SEKTION: DEPOT ---
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot_list):
    d = get_full_stock_info(t)
    if d:
        with d_cols[i % 4]:
            st.write(f"**{t}**: {d['price']:.2f}$")

st.write("---") 

# --- 6. SEKTION: EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker eingeben", value="HOOD").upper()

if t_in:
    d = get_full_stock_info(t_in)
    if d and d['dates']:
        if d['has_e']: st.error(f"⚠️ ACHTUNG: Earnings am {d['e_date']}!")
        st.write(f"Aktueller Kurs: **{d['price']:.2f}$**")
        d_sel = st.selectbox("Laufzeit wählen", d['dates'])
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
        T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
        df = chain[chain['strike'] < d['price']].sort_values('strike', ascending=False).head(6) if mode == "put" else chain[chain['strike'] > d['price']].sort_values('strike', ascending=True).head(6)
        
        for _, opt in df.iterrows():
            delta = calculate_bsm_delta(d['price'], opt['strike'], T, opt['impliedVolatility'] or 0.4, mode)
            with st.expander(f"Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$"):
                st.write(f"💰 **Cash-Einnahme: {opt['bid']*100:.0f}$**")
                st.write(f"📊 **Prob. OTM: {(1-abs(delta))*100:.1f}%** | Puffer: {(abs(opt['strike']-d['price'])/d['price'])*100:.1f}%")
