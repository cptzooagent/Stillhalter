import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import concurrent.futures
import time
import requests

# --- 1. SETUP & SESSION MANAGEMENT ---
st.set_page_config(page_title="CapTrader AI Pro Scanner", layout="wide")

@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 2. MATHEMATISCHE FUNKTIONEN (CORE) ---
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

def get_stock_data_full(symbol, session=None):
    try:
        tk = yf.Ticker(symbol, session=session)
        hist = tk.history(period="250d")
        if hist.empty: return None
        
        price = hist['Close'].iloc[-1]
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        # Pivot Punkte (S2/R2)
        last_day = hist.iloc[-2]
        p_d = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
        s2_d = p_d - (last_day['High'] - last_day['Low'])
        r2_d = p_d + (last_day['High'] - last_day['Low'])
        
        # Weekly Pivots (simuliert aus Daily)
        hist_w = hist.resample('W').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        last_w = hist_w.iloc[-2]
        p_w = (last_w['High'] + last_w['Low'] + last_w['Close']) / 3
        s2_w = p_w - (last_w['High'] - last_w['Low'])
        
        return {
            'price': price, 'rsi': rsi, 'uptrend': price > sma_200,
            'pivots': {'S2': s2_d, 'R2': r2_d, 'W_S2': s2_w},
            'dates': tk.options, 'info': tk.info
        }
    except: return None

@st.cache_data(ttl=86400)
def get_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN"]


# --- 3. SCANNER LOGIK & UI ---
with st.sidebar:
    st.header("🛡️ Filter-Einstellungen")
    otm_puffer = st.slider("OTM Puffer (%)", 5, 25, 12)
    min_yield = st.number_input("Min. Yield p.a. (%)", 5, 100, 15)
    test_mode = st.checkbox("Simulations-Modus", value=True)

st.title("🚀 CapTrader AI: Smart Option Cockpit")
session = get_session()

if st.button("🔍 Markt jetzt scannen", use_container_width=True):
    tickers = ["NVDA", "TSLA", "AMD", "PLTR", "MU", "COIN", "MSTR", "SE", "HOOD", "CRWD"] if test_mode else get_watchlist()
    results = []
    prog = st.progress(0)
    
    def scan_stock(s):
        data = get_stock_data_full(s, session)
        if not data or not data['dates']: return None
        try:
            tk = yf.Ticker(s, session=session)
            heute = datetime.now()
            valid = [d for d in data['dates'] if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
            if not valid: return None
            
            chain = tk.option_chain(valid[0]).puts
            target_strike = data['price'] * (1 - (otm_puffer/100))
            opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if opts.empty: return None
            o = opts.iloc[0]
            mid = (o['bid'] + o['ask']) / 2
            days = (datetime.strptime(valid[0], '%Y-%m-%d') - heute).days
            y_pa = (mid / o['strike']) * (365/max(1, days)) * 100
            
            if y_pa >= min_yield:
                return {
                    'symbol': s, 'price': data['price'], 'y_pa': y_pa, 'strike': o['strike'],
                    'rsi': data['rsi'], 'puffer': otm_puffer, 'tage': days, 
                    'uptrend': data['uptrend'], 's2_w': data['pivots']['W_S2']
                }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_stock, t): t for t in tickers}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: results.append(res)
            prog.progress((i+1)/len(tickers))
    
    st.session_state.scan_results = sorted(results, key=lambda x: x['y_pa'], reverse=True)

# --- 4. DISPLAY: DIE KACHELN ---
if 'scan_results' in st.session_state and st.session_state.scan_results:
    st.markdown("### 🎯 Top-Opportunitäten")
    cols = st.columns(3)
    for idx, res in enumerate(st.session_state.scan_results):
        with cols[idx % 3]:
            # Optische Ampeln
            rsi_col = "#10b981" if res['rsi'] < 35 else "#ef4444" if res['rsi'] > 70 else "#3b82f6"
            pivot_dist = (res['price'] - res['s2_w']) / res['price'] * 100
            
            st.markdown(f"""
                <div style="background: white; border-radius: 15px; padding: 20px; border-top: 5px solid {rsi_col}; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 20px; color: #1f2937;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 1.5em; font-weight: 800;">{res['symbol']}</span>
                        <span style="color: {'#10b981' if res['uptrend'] else '#f59e0b'}; font-weight: 700;">
                            {'📈 Trend' if res['uptrend'] else '📉 Dip'}
                        </span>
                    </div>
                    <div style="margin: 15px 0;">
                        <small style="color: #6b7280;">RENDITE P.A.</small>
                        <div style="font-size: 2em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; background: #f9fafb; padding: 10px; border-radius: 10px;">
                        <div><small>Strike</small><br><b>{res['strike']:.1f}$</b></div>
                        <div><small>Puffer</small><br><b>{res['puffer']}%</b></div>
                        <div><small>Tage</small><br><b>{res['tage']}d</b></div>
                    </div>
                    <div style="margin-top: 15px; font-size: 0.85em;">
                        📍 S2 Weekly: <b>{res['s2_w']:.1f}$</b> ({pivot_dist:.1f}% dist)
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 5. DEPOT-MANAGER (PIVOT-REPARATUR) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Pivot-Reparatur")
st.info("Hier werden deine Bestände gegen die wöchentlichen S2-Punkte geprüft.")
# (Hier kannst du die Depot-Tabelle aus dem vorherigen Schritt einfügen)
