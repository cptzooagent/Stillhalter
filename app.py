import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Pro Stillhalter Dashboard", layout="wide")

# API Keys aus den Streamlit Secrets
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- DATA FUNCTIONS (MIT CACHING ZUR LIMIT-SCHONUNG) ---

@st.cache_data(ttl=600)  # Speichert Marktdaten für 10 Minuten
def get_market_metrics():
    data = {"VIX": 15.0, "BTC": 0.0, "SP500": 0.0, "NASDAQ": 0.0, "SP_CHG": 0.0, "NAS_CHG": 0.0}
    try:
        # VIX via MarketData
        r_vix = requests.get(f"https://api.marketdata.app/v1/indices/quotes/VIX/?token={MD_KEY}").json()
        if r_vix.get('s') == 'ok': data["VIX"] = r_vix['last'][0]
        
        # Bitcoin & Indizes via Finnhub (höheres Limit)
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

@st.cache_data(ttl=3600) # Laufzeiten ändern sich selten -> 1 Std Cache
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

# --- UI STRUKTUR ---

# 1. HEADER METRIKEN
m = get_market_metrics()
st.title("🛡️ CapTrader Pro Scanner")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX", f"{m['VIX']:.2f}", "🔥 Hoch" if m['VIX'] > 25 else "🟢 Normal", delta_color="inverse")
    c2.metric("Bitcoin", f"{m['BTC']:,.0f} $")
    c3.metric("S&P 500", f"{m['SP500']:,.0f}", f"{m['SP_CHG']:.2f}%")
    c4.metric("Nasdaq", f"{m['NASDAQ']:,.0f}", f"{m['NAS_CHG']:.2f}%")

# 2. PORTFOLIO & REPAIR
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
with f2: ticker = st.text_input("Ticker eingeben (z.B. HOOD)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        dates = get_all_expirations(ticker)
        if dates:
            sel_date = st.selectbox("Laufzeit wählen", dates)
            df = get_chain_for_date(ticker, sel_date, side)
            if df is not None and not df.empty:
                # Filter OTM & Sortierung
                df = df[df['strike'] < price] if side == "put" else df[df['strike'] > price]
                df = df.sort_values('strike', ascending=(side == "call"))
                
                for _, row in df.head(8).iterrows():
                    pop = (1 - abs(row['delta'])) * 100
                    color = "🟢" if abs(row['delta']) < 0.15 else "🟡"
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}%"):
                        st.write(f"Prämie: **{row['mid']:.2f}$** | Delta: {row['delta']:.2f} | IV: {row['iv']*100:.1f}%")
        else:
            st.error("Keine Laufzeiten gefunden. API-Limit für heute erreicht?")

st.divider()

# 4. HIGH-IV SCANNER
st.subheader("💎 Top 10 High-IV Put Gelegenheiten")
if st.button("🔥 Markt-Scan starten"):
    watchlist = ["TSLA", "NVDA", "AMD", "COIN", "MARA", "PLTR", "AFRM", "SQ", "HOOD", "SOFI"]
    opps = []
    with st.spinner("Scanne High-IV Werte..."):
        for t in watchlist:
            exp = get_all_expirations(t)
            if exp:
                # Wähle Laufzeit nahe 30-45 Tage
                target = next((d for d in exp if 25 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 55), exp[0])
                df = get_chain_for_date(t, target, "put")
                if df is not None:
                    df['diff'] = (df['delta'].abs() - 0.15).abs()
                    best = df.sort_values('diff').iloc[0].to_dict()
                    best.update({'ticker': t, 'yield': (best['mid']/best['strike'])*12*100})
                    opps.append(best)
    
    if opps:
        res_df = pd.DataFrame(opps).sort_values('yield', ascending=False)
        cols = st.columns(5)
        for idx, row in res_df.iterrows():
            with cols[idx % 5]:
                st.metric(row['ticker'], f"{row['yield']:.1f}% p.a.", f"Strike: {row['strike']}")
