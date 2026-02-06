import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

# --- 1. MATHE & HELFER ---
def calc_delta(S, K, T, sigma, opt='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt == 'call' else norm.cdf(d1) - 1

@st.cache_data(ttl=900)
def get_stock_data(s):
    try:
        tk = yf.Ticker(s)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn, ed = False, None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            edt = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            if 0 <= (edt - datetime.now()).days <= 14:
                earn, ed = True, edt.strftime('%d.%m.')
        return {"p": price, "d": dates, "e": earn, "ed": ed}
    except: return None

# --- 2. UI SETUP ---
st.set_page_config(page_title="CapTrader AI", layout="wide")
st.sidebar.header("🛡️ Filter")
prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
m_delta, m_y = (100-prob)/100, st.sidebar.number_input("Mindestrendite p.a. (%)", value=10)
st.title("🛡️ CapTrader AI Market Scanner")

# --- 3. SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    wl = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI", "MSTR", "AI"]
    res = []
    prog = st.progress(0)
    for i, t in enumerate(wl):
        prog.progress((i+1)/len(wl))
        data = get_stock_data(t)
        if data and data['d']:
            try:
                tk = yf.Ticker(t)
                td = min(data['d'], key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d')-datetime.now()).days-30))
                ch = tk.option_chain(td).puts
                T = (datetime.strptime(td, '%Y-%m-%d')-datetime.now()).days/365
                ch['dv'] = ch.apply(lambda r: calc_delta(data['p'], r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                safe = ch[ch['dv'].abs() <= m_delta].copy()
                if not safe.empty:
                    best = safe.sort_values('dv', ascending=False).iloc[0]
                    y = (best['bid']/best['strike'])*(365/max(1, (datetime.strptime(td, '%Y-%m-%d')-datetime.now()).days))*100
                    if y >= m_y:
                        res.append({'t':t, 'y':y, 's':best['strike'], 'b':best['bid'], 'p':(abs(best['strike']-data['p'])/data['p'])*100, 'e':data['e'], 'ed':data['ed']})
            except: continue
    if res:
        cols = st.columns(4)
        for idx, r in enumerate(pd.DataFrame(res).sort_values('p', ascending=False).head(8).to_dict('records')):
            with cols[idx%4]:
                st.markdown(f"### {r['t']}")
                if r['e']: st.caption(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"💰 Bid: **{r['b']:.2f}$** | Yield: **{r['y']:.1f}%**")
                st.write(f"🎯 Strike: **{r['s']:.1f}$**")

st.write("---")

# --- 4. DEPOT ---
st.subheader("💼 Depot-Status")
dep_list = ["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(dep_list):
    d = get_stock_data(t)
    if d:
        with d_cols[i%4]: st.write(f"**{t}**: {d['p']:.2f}$")

st.write("---")

# --- 5. EINZEL-CHECK ---
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: m = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: ti = st.text_input("Ticker", "HOOD").upper()
if ti:
    d = get_stock_data(ti)
    if d and d['d']:
        if d['e']: st.warning(f"⚠️ Earnings am {d['ed']}!")
        sel_d = st.selectbox("Laufzeit", d['d'])
        tk = yf.Ticker(ti)
        ch = tk.option_chain(sel_d).puts if m=="put" else tk.option_chain(sel_d).calls
        T = (datetime.strptime(sel_d, '%Y-%m-%d')-datetime.now()).days/365
        df = ch[ch['strike'] < d['p']].sort_values('strike', ascending=False).head(6) if m=="put" else ch[ch['strike'] > d['p']].sort_values('strike', ascending=True).head(6)
        for _, o in df.iterrows():
            dl = calc_delta(d['p'], o['strike'], T, o['impliedVolatility'] or 0.4, m)
            with st.expander(f"Strike {o['strike']:.1f}$ | Prämie: {o['bid']:.2f}$"):
                st.write(f"💰 **Einnahme: {o['bid']*100:.0f}$** | OTM: **{(1-abs(dl))*100:.1f}%**")
                st.write(f"🎯 Puffer: {(abs(o['strike']-d['p'])/d['p'])*100:.1f}% | Delta: {abs(dl):.2f}")
