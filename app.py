import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP & PROFESSIONAL STYLING ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# Custom CSS für echtes Dashboard-Feeling (kompatibel mit älteren Versionen)
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MATHE & LOGIK ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-ENGINE ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_date, earn_str = None, ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0]
                earn_str = earn_date.strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, earn_date
    except:
        return None, [], "", 50, None

# --- SIDEBAR (Kompatibilitäts-Fix) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1534/1534003.png", width=80)
    st.header("Konfiguration")
    target_prob = st.slider("🛡️ Sicherheit (OTM %)", 70, 98, 85)
    max_delta = (100 - target_prob) / 100
    min_yield_pa = st.number_input("💵 Min. Rendite p.a. (%)", value=15)
    # Fix für Bild 7: checkbox statt toggle
    sort_by_rsi = st.checkbox("🔄 Nach RSI sortieren", value=False)
    st.divider()
    st.info("Scanner aktiv: S&P 500 & Nasdaq-100")

# --- HAUPTBEREICH ---
st.markdown('<p class="main-title">🛡️ CapTrader AI Market Scanner</p>', unsafe_allow_html=True)

# SCANNER SEKTION
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "PLTR", "HOOD", "AFRM", "MSTR"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'bid': best['bid'], 
                                        'puffer': (abs(best['strike'] - price) / price) * 100, 'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi})
            except: continue

    if results:
        df_res = pd.DataFrame(results)
        opp_df = df_res.sort_values('rsi' if sort_by_rsi else 'yield', ascending=False).head(12)
        cols = st.columns(3)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 3]:
                st.metric(label=f"💰 {row['ticker']}", value=f"{row['yield']:.1f}% p.a.", delta=f"RSI: {row['rsi']:.0f}")
                with st.expander("Details anzeigen"):
                    st.write(f"🎯 **Strike:** {row['strike']:.1f}$")
                    st.write(f"🛡️ **Delta:** {row['delta']:.2f}")
                    st.write(f"📉 **Puffer:** {row['puffer']:.1f}%")
                    if row['earn']: st.warning(f"Earnings: {row['earn']}")
    else:
        st.warning("Keine Treffer gefunden.")

# DEPOT SEKTION
st.markdown("### 💼 Smart Depot-Manager")
depot_list = [{"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "NVDA", "Einstand": 115.0}]

d_cols = st.columns(3)
for i, item in enumerate(depot_list):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i % 3]:
            # Kompaktes Karten-Design
            with st.expander(f"{item['Ticker']} ({perf:+.1f}%)", expanded=True):
                # Earnings Check (Fix für Bild 4)
                if earn_dt is not None:
                    try:
                        days = (earn_dt.replace(tzinfo=None) - datetime.now().replace(tzinfo=None)).days
                        if 0 <= days <= 5: st.error(f"🚨 Earnings in {days} Tagen!")
                    except: pass
                
                c1, c2 = st.columns(2)
                c1.metric("Kurs", f"{price:.1f}$")
                c2.metric("RSI", f"{rsi:.0f}")
                
                # Handlungsanweisung
                if rsi > 65: st.success("🎯 Tipp: Call verkaufen")
                elif rsi < 35: st.info("💎 Tipp: Hold (Oversold)")

# EINZEL-CHECK
st.markdown("### 🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: check_mode = st.radio("Typ", ["put", "call"], horizontal=True) # horizontal=True ist sicherer als segmented_control
with c2: check_ticker = st.text_input("Symbol", value="NVDA").upper()

if check_ticker:
    price, dates, earn, rsi, _ = get_stock_data_full(check_ticker)
    if price and dates:
        st.info(f"Aktueller Kurs: {price:.2f}$ | RSI: {rsi:.0f}")
        d_sel = st.selectbox("Laufzeit", dates)
        # ... Rest der Einzelcheck-Logik bleibt gleich ...
