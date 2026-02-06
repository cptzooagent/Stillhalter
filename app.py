import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")
POLY_KEY = st.secrets.get("POLYGON_KEY")

# --- DATA FUNCTIONS (CACHED) ---
@st.cache_data(ttl=600)
def get_market_metrics():
    """Holt Marktübersicht von Finnhub (Bitcoin, S&P, Nasdaq)"""
    data = {"VIX": 15.0, "BTC": 0.0, "SP500": 0.0, "NASDAQ": 0.0, "SP_CHG": 0.0, "NAS_CHG": 0.0}
    try:
        # VIX von MarketData
        r_vix = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}").json()
        if r_vix.get('s') == 'ok': data["VIX"] = r_vix['last'][0]
        
        # Bitcoin & ETFs von Finnhub
        for name, sym in [("BTC", "BINANCE:BTCUSDT"), ("SP500", "SPY"), ("NASDAQ", "QQQ")]:
            rf = requests.get(f'https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}').json()
            if rf.get('c'):
                if name == "BTC": data["BTC"] = rf['c']
                elif name == "SP500": 
                    data["SP500"] = rf['c'] * 10
                    data["SP_CHG"] = rf['dp']
                elif name == "NASDAQ": 
                    data["NASDAQ"] = rf['c'] * 40
                    data["NAS_CHG"] = rf['dp']
    except: pass
    return data

@st.cache_data(ttl=900)
def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

@st.cache_data(ttl=3600)
def get_all_expirations(symbol):
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        return sorted(r.get('expirations', [])) if r.get('s') == 'ok' else []
    except: return []

@st.cache_data(ttl=600)
def get_chain_for_date(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok':
            return pd.DataFrame({
                'strike': r['strike'], 'mid': r['mid'], 
                'delta': r.get('delta', [0.0]*len(r['strike'])), 
                'iv': r.get('iv', [0.0]*len(r['strike']))
            })
    except: return None

# --- UI START ---
st.title("🛡️ CapTrader Pro Scanner")

# 1. TOP METRICS BAR
m = get_market_metrics()
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX", f"{m['VIX']:.2f}", "🔥 Panik" if m['VIX'] > 25 else "🟢 Ruhig", delta_color="inverse")
    c2.metric("Bitcoin", f"{m['BTC']:,.0f} $")
    c3.metric("S&P 500", f"{m['SP500']:,.0f}", f"{m['SP_CHG']:.2f}%")
    c4.metric("Nasdaq", f"{m['NASDAQ']:,.0f}", f"{m['NAS_CHG']:.2f}%")

# 2. PORTFOLIO REPAIR
st.subheader("💼 Portfolio Repair-Status")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
        {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "TTD", "Einstand": 102.0}
    ])

c_tab, c_status = st.columns([1, 1.2])
with c_tab:
    with st.expander("Bestände editieren"):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

with c_status:
    for _, row in st.session_state.portfolio.iterrows():
        curr = get_live_price(row['Ticker'])
        if curr:
            diff = (curr/row['Einstand'] - 1) * 100
            icon, stat = ("🟢", "OK") if diff >= 0 else ("🟡", "REPAIR") if diff > -20 else ("🔵", "DEEP REPAIR")
            st.write(f"{icon} **{row['Ticker']}**: {curr:.2f}$ ({diff:.1f}%) → `{stat}`")

st.divider()

# 3. OPTIONS-FINDER
st.subheader("🔍 Options-Finder")
f1, f2 = st.columns([1, 2])
with f1: side = st.radio("Typ", ["put", "call"], horizontal=True)
with f2: ticker = st.text_input("Ticker für Detail-Scan").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        dates = get_all_expirations(ticker)
        if dates:
            sel_date = st.selectbox("Laufzeit wählen", dates)
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                for _, row in df.head(8).iterrows():
                    d_abs = abs(row['delta'])
                    pop = (1 - d_abs) * 100
                    color = "🟢" if d_abs < 0.15 else "🟡" if d_abs < 0.25 else "🔴"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {d_abs:.2f} | Chance: {pop:.0f}%"):
                        st.write(f"Prämie: **{row['mid']:.2f}$** | IV: **{row['iv']*100:.1f}%**")

st.divider()

# 4. TOP 10 HIGH IV LISTE
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta 0.15)")
watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI"]
if st.button("🔥 High-IV Scan starten"):
    opps = []
    with st.spinner("Scanne Märkte..."):
        for t in watchlist:
            dates = get_all_expirations(t)
            if dates:
                target = next((d for d in dates if 25 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 50), dates[0])
                df = get_chain_for_date(t, target, "put")
                if df is not None:
                    df['diff'] = (df['delta'].abs() - 0.15).abs()
                    best = df.sort_values('diff').iloc[0].to_dict()
                    days = (datetime.strptime(target, '%Y-%m-%d') - datetime.now()).days
                    best.update({'ticker': t, 'yield': (best['mid']/best['strike'])*(365/days)*100, 'days': days})
                    opps.append(best)
    
    if opps:
        opp_df = pd.DataFrame(opps).sort_values('yield', ascending=False)
        cols = st.columns(5)
        for idx, row in opp_df.iterrows():
            with cols[idx % 5]:
                with st.container(border=True):
                    st.markdown(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%", f"{row['mid']:.2f}$")
                    st.caption(f"Strike: {row['strike']} | {row['days']} Tage")
