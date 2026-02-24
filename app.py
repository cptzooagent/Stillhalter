import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
import fear_and_greed
import requests

# --- 0. SESSION & ANTI-BLOCK SETUP ---
@st.cache_resource
def get_yf_session():
    """Erstellt eine persistente Session gegen Yahoo HTTP 401 Fehler."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    })
    return session

yf_session = get_yf_session()

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# CSS für maximale Breite
st.markdown("""
<style>
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; }
    [data-testid="stHorizontalBlock"] { gap: 10px !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. CORE LOGIK ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_pivots(symbol):
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        hist_d = tk.history(period="5d")
        if len(hist_d) < 2: return None
        ld = hist_d.iloc[-2]
        p = (ld['High'] + ld['Low'] + ld['Close']) / 3
        return {"P": p, "S1": (2*p)-ld['High'], "S2": p-(ld['High']-ld['Low']), "R2": p+(ld['High']-ld['Low'])}
    except: return None

def get_openclaw_analysis(symbol):
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        news = tk.news
        if not news: return "Neutral", "Keine aktuellen News.", 0.5
        blob = str(news).lower()
        score = 0.5
        for w in ['growth', 'beat', 'buy', 'ai', 'up']: 
            if w in blob: score += 0.07
        for w in ['miss', 'down', 'sell', 'risk']: 
            if w in blob: score -= 0.07
        score = max(0.1, min(0.9, score))
        status = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
        return status, f"OpenClaw: {news[0]['title'][:80]}...", score
    except: return "N/A", "System-Reset...", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL"]

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol, session=yf_session)
        hist = tk.history(period="150d")
        if hist.empty: return None
        price = hist['Close'].iloc[-1]
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist)>=200 else hist['Close'].mean()
        earn = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, list(tk.options), earn, rsi, (price > sma200), tk.info
    except: return None

# --- 2. DASHBOARD ---
def get_market_data():
    try:
        n = yf.Ticker("^NDX", session=yf_session); v = yf.Ticker("^VIX", session=yf_session)
        hn = n.history(period="1mo"); hv = v.history(period="1d")
        cp = hn['Close'].iloc[-1]; sma = hn['Close'].rolling(20).mean().iloc[-1]
        return cp, calculate_rsi(hn['Close']).iloc[-1], ((cp-sma)/sma)*100, hv['Close'].iloc[-1]
    except: return 0, 50, 0, 20

cp_n, rsi_n, dist_n, vix = get_market_data()
stock_fg = fear_and_greed.get_index().value rescue 45

m_color = "#27ae60" if vix < 22 else "#e74c3c"
st.markdown(f'<div style="background:{m_color};color:white;padding:15px;border-radius:12px;text-align:center;"><h3>Markt-Status: {"STABIL" if vix < 22 else "VOLATIL"}</h3></div>', unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Konfiguration")
    p_puffer = st.slider("Puffer (%)", 3, 25, 15)
    p_yield = st.number_input("Min. % p.a.", 0, 100, 12)
    p_price = st.slider("Preis ($)", 0, 1000, (50, 600))
    p_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 20)
    test_mode = st.checkbox("Simulations-Modus", value=False)

# --- 4. SCANNER ---
if 'res' not in st.session_state: st.session_state.res = []

if st.button("🚀 Profi-Scan starten"):
    heute = datetime.now()
    watchlist = ["NVDA", "TSLA", "AAPL", "AMD", "COIN", "MSTR", "PLTR"] if test_mode else get_combined_watchlist()
    found = []

    def process(s):
        try:
            time.sleep(0.4)
            d = get_stock_data_full(s)
            if not d or not (p_price[0] <= d[0] <= p_price[1]): return None
            price, opts, earn, rsi, trend, info = d
            if info.get('marketCap', 0) < p_cap*1e9: return None
            
            valid_dates = [o for o in opts if 10 <= (datetime.strptime(o, '%Y-%m-%d')-heute).days <= 40]
            if not valid_dates: return None
            
            tk = yf.Ticker(s, session=yf_session)
            chain = tk.option_chain(valid_dates[0]).puts
            strike = price * (1 - p_puffer/100)
            target = chain[chain['strike'] <= strike].sort_values('strike', ascending=False)
            if target.empty: return None
            o = target.iloc[0]
            
            bid, ask = o['bid'], o['ask']
            if bid <= 0.05: return None
            fair = (bid + ask) / 2
            days = (datetime.strptime(valid_dates[0], '%Y-%m-%d')-heute).days
            y_pa = (fair / o['strike']) * (365 / days) * 100
            
            if y_pa < p_yield: return None
            
            oc_status, oc_text, oc_score = get_openclaw_analysis(s)
            
            return {
                's': s, 'p': price, 'y': y_pa, 'st': o['strike'], 'pu': ((price-o['strike'])/price)*100,
                'rsi': rsi, 'earn': earn, 'days': days, 'trend': trend, 'oc': oc_text,
                'cap': info.get('marketCap', 0)/1e9, 'iv': o.get('impliedVolatility', 0)*100
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(process, watchlist))
        st.session_state.res = [r for r in results if r]

# --- 5. RENDER CARDS ---
if st.session_state.res:
    cols = st.columns(4)
    for idx, r in enumerate(st.session_state.res):
        with cols[idx % 4]:
            # HTML CODE GANZ LINKS BÜNDIG IM CODE
            html = f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.1em; font-weight: 800; color: #111827;">{r['s']}</span>
<span style="font-size: 0.7em; font-weight: 700; color: #3b82f6; background: #3b82f610; padding: 2px 8px; border-radius: 6px;">{"TREND" if r['trend'] else "DIP"}</span>
</div>
<div style="margin: 10px 0;">
<div style="font-size: 0.6em; color: #6b7280; font-weight: 600;">RENDITE P.A.</div>
<div style="font-size: 1.8em; font-weight: 900; color: #111827;">{r['y']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;">
<div style="font-size: 0.55em; color: #6b7280;">Strike</div>
<div style="font-size: 0.85em; font-weight: 700;">{r['st']:.1f}$</div>
</div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;">
<div style="font-size: 0.55em; color: #6b7280;">Puffer</div>
<div style="font-size: 0.85em; font-weight: 700;">{r['pu']:.1f}%</div>
</div>
</div>
<div style="font-size: 0.65em; color: #4b5563; background: #f9fafb; padding: 8px; border-radius: 8px; border: 1px solid #f3f4f6;">
{r['oc']}
</div>
<div style="margin-top: 10px; display: flex; justify-content: space-between; font-size: 0.6em; color: #9ca3af;">
<span>RSI: {r['rsi']:.0f}</span>
<span>🗓️ {r['earn'] if r['earn'] else "N/A"}</span>
<span>⏳ {r['days']} Tage</span>
</div>
</div>
"""
            st.markdown(html, unsafe_allow_html=True)
