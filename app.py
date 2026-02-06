import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    T = max(T, 0.0001) 
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# --- 2. SICHERHEITS-INDIKATOREN (KI-Logik) ---
def get_safety_metrics(tk):
    """Berechnet RSI und Trend für maximale Sicherheit."""
    try:
        hist = tk.history(period="3mo")
        if len(hist) < 20: return 50, True
        
        # RSI Berechnung (14 Tage)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Trend: Kurs über/unter 50-Tage-Linie
        sma50 = hist['Close'].tail(50).mean()
        current_price = hist['Close'].iloc[-1]
        trend_ok = current_price > (sma50 * 0.95) # Max 5% unter SMA50 erlaubt
        
        return round(rsi, 1), trend_ok
    except:
        return 50, True

# --- 3. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=3600)
def get_combined_watchlist():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "NFLX", 
        "AMD", "INTC", "QCOM", "AMAT", "TXN", "MU", "ISRG", "LRCX", "PANW", "SNPS",
        "LLY", "V", "MA", "JPM", "WMT", "XOM", "UNH", "PG", "ORCL", "COST", 
        "ABBV", "BAC", "KO", "PEP", "CRM", "WFC", "DIS", "CAT", "AXP", "IBM",
        "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR",
        "SMCI", "BKNG", "DE", "GS", "MS", "BA", "SBUX", "UBER", "ABNB"
    ]

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_info = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_info = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_info, tk
    except: return None, [], "", None

# --- UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie & Sicherheit")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
rsi_min = st.sidebar.slider("Minimum RSI (kein freier Fall)", 20, 40, 30)

st.sidebar.subheader("💰 Preis-Filter")
min_stock_p = st.sidebar.number_input("Mindestkurs ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs ($)", value=600)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Guard Pro")

# SEKTION 1: SCANNER
if st.button("🚀 High-Safety Scan starten"):
    watchlist = get_combined_watchlist()
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Checke Sicherheit für {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, tk_obj = get_stock_basics(t)
        
        if price and min_stock_p <= price <= max_stock_p and dates:
            rsi, trend_ok = get_safety_metrics(tk_obj)
            
            # KI-SICHERHEITS-FILTER
            if rsi < rsi_min or not trend_ok:
                continue 
                
            try:
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk_obj.option_chain(target_date).puts
                T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                max_delta = (100 - target_prob) / 100
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({
                            'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 
                            'bid': best['bid'], 'rsi': rsi, 'price': price, 
                            'earn': earn, 'capital': best['strike'] * 100,
                            'puffer': (abs(best['strike'] - price) / price) * 100
                        })
            except: continue

    status.text("Sicherheits-Scan abgeschlossen!")
    if results:
        opp_df = pd.DataFrame(results).sort_values('yield', ascending=False).head(12)
        cols = st.columns(4)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"Strike: **{row['strike']:.1f}$** (Puff: {row['puffer']:.1f}%)")
                st.write(f"RSI: `{row['rsi']}` | ER: `{row['earn'] if row['earn'] else 'n/a'}`")
                st.info(f"💼 Kapital: {row['capital']:,.0f}$")
    else: st.warning("Keine sicheren Treffer gefunden. RSI oder Rendite-Filter anpassen.")

st.write("---") 

# SEKTION 2: DEPOT
st.subheader("💼 Depot-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]
p_cols = st.columns(4)
for i, item in enumerate(depot_data):
    price, _, earn, _ = get_stock_basics(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        icon = "🟢" if diff >= 0 else "🟡" if diff > -20 else "🔵"
        with p_cols[i % 4]:
            st.write(f"{icon} **{item['Ticker']}**: {price:.2f}$ ({diff:.1f}%)")

st.write("---") 

# SEKTION 3: EINZEL-CHECK
st.subheader("🔍 Experten Einzel-Check")
t_in = st.text_input("Ticker für Detail-Analyse", value="NVDA").upper()
if t_in:
    price, dates, earn, tk_obj = get_stock_basics(t_in)
    if price and dates:
        rsi, trend_ok = get_safety_metrics(tk_obj)
        col1, col2, col3 = st.columns(3)
        col1.metric("Kurs", f"{price:.2f}$")
        col2.metric("RSI (14d)", rsi, delta="Überverkauft" if rsi < 30 else "Neutral")
        col3.metric("Trend", "Intakt" if trend_ok else "Abwärts")
