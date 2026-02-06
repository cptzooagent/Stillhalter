import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import random

def calc_delta(S, K, T, sigma, opt='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) if 'r' in locals() else (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt == 'call' else norm.cdf(d1) - 1

@st.cache_data(ttl=86400)
def get_wl():
    b = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI"]
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt", header=None)
        return list(set(b + df[0].head(80).tolist()))
    except: return b

@st.cache_data(ttl=900)
def get_data(s):
    try:
        tk = yf.Ticker(s)
        inf = tk.fast_info
        earn, ed = False, None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            edt = tk.calendar['Earnings Date'][0].replace(tzinfo=None)
            if 0 <= (edt - datetime.now()).days <= 14: earn, ed = True, edt.strftime('%d.%m.')
        return {"p": inf['last_price'], "d": list(tk.options), "e": earn, "ed": ed}
    except: return None

st.set_page_config(layout="wide")
st.sidebar.header("Filter")
prob = st.sidebar.slider("OTM %", 70, 98, 85)
m_delta, m_y = (100-prob)/100, st.sidebar.number_input("Min Yield %", value=10)

st.title("🛡️ CapTrader AI Market Scanner")

if st.button("🚀 Scan starten"):
    res = []
    wl = random.sample(get_wl(), 40)
    p_bar = st.progress(0)
    for i, t in enumerate(wl):
        p_bar.progress((i+1)/len(wl))
        d = get_data(t)
        if d and d['d']:
            try:
                tk = yf.Ticker(t)
                td = min(d['d'], key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d')-datetime.now()).days-30))
                ch = tk.option_chain(td).puts
                T = (datetime.strptime(td, '%Y-%m-%d')-datetime.now()).days/365
                ch['dv'] = ch.apply(lambda r: calc_delta(d['p'], r['strike'], T, r['impliedVolatility'] or 0.5), axis=1)
                s_o = ch[ch['dv'].abs() <= m_delta].copy()
                if not s_o.empty:
                    b = s_o.sort_values('dv', ascending=False).iloc[0]
                    days = max(1, (datetime.strptime(td, '%Y-%m-%d')-datetime.now()).days)
                    y = (b['bid']/b['strike'])*(365/days)*100
                    if y >= m_y: res.append({'t':t,'y':y,'s':b['strike'],'b':b['bid'],'p':(abs(b['strike']-d['p'])/d['p'])*100,'e':d['e'],'ed':d['ed']})
            except: continue
    if res:
        cols = st.columns(5)
        for i, r in enumerate(pd.DataFrame(res).sort_values('p', ascending=False).head(10).to_dict('records')):
            with cols[i%5]:
                st.write(f"### {r['t']}")
                if r['e']: st.caption(f"⚠️ Earnings: {r['ed']}")
                st.metric("Puffer", f"{r['p']:.1f}%")
                st.write(f"**Bid: {r['b']:.2f}$ | Yield: {r['y']:.1f}%**")
                st.write(f"Strike: {r['s']:.1f}$")

st.write("---")
st.subheader("💼 Depot")
for i, t in enumerate(["AFRM", "ELF", "ETSY", "GTLB", "HOOD", "NVO", "SE", "TTD"]):
    d = get_data(t)
    if d: st.sidebar.write(f"{t}: {d['p']:.2f}$") # Depot in Sidebar für Platzersparnis

st.write("---")
st.subheader("🔍 Check")
c1, c2 = st.columns([1,2])
with c1: m = st.radio("Typ", ["put", "call"])
with c2: ti = st.text_input("Ticker", "HOOD").upper()
if ti:
    d = get_data(ti)
    if d and d['d']:
        if d['e']: st.warning(f"⚠️ Earnings: {d['ed']}")
        sel_d = st.selectbox("Datum", d['d'])
        tk = yf.Ticker(ti)
        ch = tk.option_chain(sel_d).puts if m=="put" else tk.option_chain(sel_d).calls
        T = (datetime.strptime(sel_d, '%Y-%m-%d')-datetime.now()).days/365
        df = ch[ch['strike']<d['p']].sort_values('strike', ascending=False).head(6) if m=="put" else ch[ch['strike']>d['p']].sort_values('strike', ascending=True).head(6)
        for _, o in df.iterrows():
            dl = calc_delta(d['p'], o['strike'], T, o['impliedVolatility'] or 0.4, m)
            with st.expander(f"Strike {o['strike']} | Bid: {o['bid']}$"):
                st.write(f"💰 **Einnahme: {o['bid']*100:.0f}$** | OTM: **{(1-abs(dl))*100:.1f}%**")
                st.write(f"Puffer: {(abs(o['strike']-d['p'])/d['p'])*100:.1f}% | Delta: {abs(dl):.2f}")
