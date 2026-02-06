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

# --- DATA FUNCTIONS (MIT AUTO-SWITCH & CACHING) ---

@st.cache_data(ttl=600)
def get_market_metrics():
    data = {"VIX": 15.0, "BTC": 0.0, "SP500": 0.0, "NASDAQ": 0.0, "SP_CHG": 0.0, "NAS_CHG": 0.0}
    try:
        r_vix = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}").json()
        if r_vix.get('s') == 'ok': data["VIX"] = r_vix['last'][0]
        
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
    # 1. Versuch: MarketData
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        if r.get('s') == 'ok': return sorted(r.get('expirations', []))
    except: pass
    
    # 2. Versuch: Polygon Fallback (Gibt Liste mit heutigem Datum zurück für Snapshot-Suche)
    return [datetime.now().strftime("%Y-%m-%d")]

@st.cache_data(ttl=600)
def get_chain_for_date(symbol, date_str, side):
    # 1. Versuch: MarketData
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok' and len(r.get('strike', [])) > 0:
            return pd.DataFrame({
                'strike': r['strike'], 'mid': r['mid'], 
                'delta': r.get('delta', [0.0]*len(r['strike'])), 
                'iv': r.get('iv', [0.0]*len(r['strike']))
            })
    except: pass

    # 2. Versuch: Polygon Snapshot
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLY_KEY}"
        r = requests.get(url).json()
        if r.get('status') == 'OK':
            data = []
            for res in r.get('results', []):
                if side.upper() in res['details']['ticker']:
                    data.append({
                        'strike': res['details']['strike_price'],
                        'mid': res.get('last_quote', {}).get('p', 0),
                        'delta': -0.15 if side == "put" else 0.15, # Schätzwert
                        'iv': res.get('implied_volatility', 0)
                    })
            return pd.DataFrame(data)
    except: pass
    return None

# --- UI START ---
st.title("🛡️ Pro Stillhalter Dashboard")

# 1. MARKT-STATUS
m = get_market_metrics()
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX", f"{m['VIX']:.2f}", "🔥 Angst" if m['VIX'] > 25 else "🟢 Ruhig", delta_color="inverse")
    c2.metric("Bitcoin", f"{m['BTC']:,.0f} $")
    c3.metric("S&P 500", f"{m['SP500']:,.0f}", f"{m['SP_CHG']:.2f}%")
    c4.metric("Nasdaq", f"{m['NASDAQ']:,.0f}", f"{m['NAS_CHG']:.2f}%")

# 2. PORTFOLIO STATUS
st.subheader("💼 Portfolio Repair-Status")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
        {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "TTD", "Einstand": 102.0}
    ])

c_edit, c_view = st.columns([1, 1.2])
with c_edit:
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")
with c_view:
    for _, row in st.session_state.portfolio.iterrows():
        curr = get_live_price(row['Ticker'])
        if curr:
            diff = (curr/row['Einstand'] - 1) * 100
            icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
            st.write(f"{icon} **{row['Ticker']}**: {curr:.2f}$ ({diff:.1f}%)")

st.divider()

# 3. OPTIONS-FINDER
st.subheader("🔍 Options-Finder")
f1, f2 = st.columns([1, 2])
with f1: side = st.radio("Typ", ["put", "call"], horizontal=True)
with f2: ticker = st.text_input("Ticker eingeben").strip().upper()

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
                    color = "🟢" if d_abs < 0.15 else "🟡"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        st.write(f"Prämie: **{row['mid']:.2f}$** | Delta: {row['delta']:.2f} | IV: {row['iv']*100:.1f}%")

st.divider()

# 4. TOP 10 HIGH IV LISTE
st.subheader("💎 Top 10 High-IV Put Gelegenheiten")
if st.button("🔥 High-IV Scan starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI"]
    opps = []
    with st.spinner("Scanne..."):
        for t in watchlist:
            exp = get_all_expirations(t)
            if exp:
                target = next((d for d in exp if 25 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 55), exp[0])
                df = get_chain_for_date(t, target, "put")
                if df is not None and not df.empty:
                    df['diff'] = (df['delta'].abs() - 0.15).abs()
                    best = df.sort_values('diff').iloc[0].to_dict()
                    best.update({'ticker': t, 'yield': (best['mid']/best['strike'])*12*100})
                    opps.append(best)
    if opps:
        res_df = pd.DataFrame(opps).sort_values('yield', ascending=False)
        cols = st.columns(5)
        for idx, row in res_df.iterrows():
            with cols[idx % 5]:
                with st.container(border=True):
                    st.markdown(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%", f"{row['mid']:.2f}$")
