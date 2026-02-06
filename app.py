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
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- 2. DATEN-ENGINE (FIXED CACHING) ---
@st.cache_data(ttl=900)
def get_clean_data(symbol):
    """Speichert nur einfache Datentypen im Cache, um Fehler zu vermeiden."""
    try:
        tk = yf.Ticker(symbol)
        inf = tk.fast_info
        
        # Earnings Check
        has_e, e_dt = False, None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            edt = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            days_to = (edt - datetime.now()).days
            if 0 <= days_to <= 14:
                has_e, e_dt = True, edt.strftime('%d.%m.')
        
        return {
            "p": inf['last_price'], 
            "d": list(tk.options), 
            "has_e": has_e, 
            "e_dt": e_dt
        }
    except: return None

# --- 3. UI SETUP ---
st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")
st.sidebar.header("🛡️ Strategie-Filter")
prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
m_delta, min_y = (100 - prob) / 100, st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)

st.title("🛡️ CapTrader AI Market Scanner")

# --- 4. SEKTION: SCANNER ---
if st.button("🚀 Markt-Scan mit Earnings-Check starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI", "MSTR", "AI"]
    res = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        data = get_clean_data(t)
        if data and data['d']:
            try:
                tk = yf.Ticker(t)
                td = min(data['d'], key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                ch = tk.option_chain(td).puts
                T = (datetime.strptime(td, '%Y-%m-%d') - datetime.now()).days / 365
                ch['dv'] = ch.apply(lambda r: calculate_bsm_delta(data['p'], r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe = ch[ch['dv'].abs() <= m_delta].copy()
                if not safe.empty:
                    best = safe.sort_values('dv', ascending=False).iloc[0]
                    y = (best['bid']/best['strike']) * (365/max(1, (datetime.strptime(td, '%Y-%m-%d')-datetime.now()).days)) * 100
                    if y >= min_y:
                        res.append({'t':t, 'y':y, 's':best['strike'], 'b':best['bid'], 'p':(abs(best['strike']-data['p'])/data['p'])*100, 'e':data['has_e'], 'ed':data['e_dt']})
            except: continue
    if res:
        cols = st.columns(4)
        for idx, r in enumerate(pd.DataFrame(res).sort_values('p', ascending=False).head(8).to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {r['t']}")
                if r['e']: st.warning(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"💰 Bid: **{r['b']:.2f}$** | Yield: **{r['y']:.1f}%**")
                st.write(f"🎯 Strike: **{r['s']:.1f}$**")

st.write("---") 

# --- 5. SEKTION: DEPOT ---
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot_list):
    data = get_clean_data(t)
    if data:
        with d_cols[i % 4]:
            icon = "🔴" if data['has_e'] else "🟢"
            st.write(f"{icon} **{t}**: {data['p']:.2f}$")
            if data['has_e']: st.caption(f"Earnings am {data['e_dt']}")

st.write("---") 

# --- 6. SEKTION: EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker", value="HOOD").upper()

if t_in:
    data = get_clean_data(t_in)
    if data and data['d']:
        if data['has_e']: st.error(f"⚠️ ACHTUNG: Earnings am {data['e_dt']}!")
        st.write(f"Aktueller Kurs: **{data['p']:.2f}$**")
        sel_date = st.selectbox("Laufzeit", data['d'])
        tk = yf.Ticker(t_in)
        ch = tk.option_chain(sel_date).puts if mode == "put" else tk.option_chain(sel_date).calls
        T = (datetime.strptime(sel_date, '%Y-%m-%d') - datetime.now()).days / 365
        df = ch[ch['strike'] < data['p']].sort_values('strike', ascending=False).head(6) if mode == "put" else ch[ch['strike'] > data['p']].sort_values('strike', ascending=True).head(6)
        for _, o in df.iterrows():
            dl = calculate_bsm_delta(data['p'], o['strike'], T, o['impliedVolatility'] or 0.4, mode)
            with st.expander(f"Strike {o['strike']:.1f}$ | Prämie: {o['bid']:.2f}$"):
                st.write(f"💰 **Einnahme: {o['bid']*100:.0f}$** | OTM: **{(1-abs(dl))*100:.1f}%**")
                st.write(f"🎯 Puffer: {(abs(o['strike']-data['p'])/data['p'])*100:.1f}% | Delta: {abs(dl):.2f}")
