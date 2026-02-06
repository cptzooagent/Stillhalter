import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

# API Keys
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")
POLY_KEY = st.secrets.get("POLYGON_KEY")

# --- DATA FUNCTIONS (HYBRID & CACHED) ---

@st.cache_data(ttl=600)
def get_market_metrics():
    """Holt Marktübersicht (VIX, BTC, S&P 500, Nasdaq)"""
    data = {"VIX": 20.0, "BTC": 0.0, "SP500": 0.0, "NASDAQ": 0.0, "SP_CHG": 0.0, "NAS_CHG": 0.0}
    try:
        r_vix = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}", timeout=5).json()
        if r_vix.get('s') == 'ok': data["VIX"] = r_vix['last'][0]
        
        for name, sym in [("BTC", "BINANCE:BTCUSDT"), ("SP500", "SPY"), ("NASDAQ", "QQQ")]:
            rf = requests.get(f'https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}', timeout=5).json()
            if rf.get('c'):
                if name == "BTC": data["BTC"] = rf['c']
                elif name == "SP500": 
                    data["SP500"] = rf['c'] * 10
                    data["SP_CHG"] = rf.get('dp', 0)
                elif name == "NASDAQ": 
                    data["NASDAQ"] = rf['c'] * 40
                    data["NAS_CHG"] = rf.get('dp', 0)
    except: pass
    return data

@st.cache_data(ttl=900)
def get_live_price(symbol):
    """Echtzeit-Aktienkurs via Finnhub"""
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}', timeout=5).json()
        return float(r['c']) if r.get('c') else None
    except: return None

@st.cache_data(ttl=3600)
def get_all_expirations(symbol):
    """Holt Verfallstermine - Switcht zu Polygon bei Bedarf"""
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}", timeout=5).json()
        if r.get('s') == 'ok' and r.get('expirations'): return sorted(r.get('expirations'))
    except: pass
    
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLY_KEY}"
        r = requests.get(url, timeout=5).json()
        if r.get('status') == 'OK':
            dates = {datetime.strptime(res['details']['ticker'].split(symbol)[1][:6], "%y%m%d").strftime("%Y-%m-%d") for res in r.get('results', [])}
            return sorted(list(dates))
    except: pass
    return []

@st.cache_data(ttl=600)
def get_chain_for_date(symbol, date_str, side):
    """Holt Optionskette mit Auto-Fallback auf Polygon Snapshot"""
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params, timeout=5).json()
        if r.get('s') == 'ok' and len(r.get('strike', [])) > 0:
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0.0]*len(r['strike'])), 'iv': r.get('iv', [0.0]*len(r['strike']))})
    except: pass

    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLY_KEY}"
        r = requests.get(url, timeout=5).json()
        if r.get('status') == 'OK':
            data = []
            search_date = date_str.replace("-", "")[2:]
            for res in r.get('results', []):
                t = res['details']['ticker']
                if side.upper() in t and search_date in t:
                    data.append({
                        'strike': res['details']['strike_price'],
                        'mid': res.get('last_quote', {}).get('p', 0),
                        'delta': -0.15 if side == "put" else 0.15,
                        'iv': res.get('implied_volatility', 0)
                    })
            return pd.DataFrame(data)
    except: pass
    return None

# --- UI START ---

# 1. TOP METRICS
m = get_market_metrics()
st.title("🛡️ CapTrader Pro Dashboard")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX", f"{m['VIX']:.2f}", "🔥 Hoch" if m['VIX'] > 25 else "🟢 Ruhig", delta_color="inverse")
    c2.metric("Bitcoin", f"{m['BTC']:,.0f} $")
    c3.metric("S&P 500", f"{m['SP500']:,.0f}", f"{m['SP_CHG']:.2f}%")
    c4.metric("Nasdaq", f"{m['NASDAQ']:,.0f}", f"{m['NAS_CHG']:.2f}%")

# 2. PORTFOLIO REPAIR
st.subheader("💼 Portfolio Repair-Status")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, 
        {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
        {"Ticker": "HOOD", "Einstand": 20.0}, {"Ticker": "TTD", "Einstand": 102.0}
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
with f1: side = st.radio("Strategie", ["put", "call"], horizontal=True)
with f2: ticker = st.text_input("Ticker für Detail-Scan (z.B. HOOD)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        dates = get_all_expirations(ticker)
        if dates:
            sel_date = st.selectbox("Laufzeit wählen", dates)
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                for _, row in df.head(10).iterrows():
                    pop = (1 - abs(row['delta'])) * 100
                    color = "🟢" if abs(row['delta']) < 0.15 else "🟡"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        st.write(f"Prämie: **{row['mid']:.2f}$** | Delta: {row['delta']:.2f} | IV: {row['iv']*100:.1f}%")
        else: st.warning("Keine Laufzeiten gefunden. API Limits?")

st.divider()

# 4. HIGH IV SCANNER
st.subheader("💎 Top 10 High-IV Put Gelegenheiten (Delta 0.15)")
if st.button("🔥 Markt-Scan starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI"]
    opps = []
    progress = st.progress(0)
    for i, t in enumerate(watchlist):
        exp = get_all_expirations(t)
        if exp:
            target = next((d for d in exp if 25 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 50), exp[0])
            df = get_chain_for_date(t, target, "put")
            if df is not None and not df.empty:
                df['diff'] = (df['delta'].abs() - 0.15).abs()
                best = df.sort_values('diff').iloc[0].to_dict()
                days = (datetime.strptime(target, '%Y-%m-%d') - datetime.now()).days
                best.update({'ticker': t, 'yield': (best['mid']/best['strike'])*(365/max(1,days))*100, 'days': days})
                opps.append(best)
        progress.progress((i + 1) / len(watchlist))
    
    if opps:
        res_df = pd.DataFrame(opps).sort_values('yield', ascending=False)
        cols = st.columns(5)
        for idx, row in res_df.iterrows():
            with cols[idx % 5]:
                with st.container(border=True):
                    st.markdown(f"**{row['ticker']}**")
                    st.metric("Yield p.a.", f"{row['yield']:.1f}%", f"{row['mid']:.2f}$")
                    st.caption(f"Strike: {row['strike']} | {row['days']} Tage")
