import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    T = max(T, 0.0001) 
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- 2. DATEN-FUNKTIONEN (Sicherheits-Check bei jedem Schritt) ---
@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        info = tk.fast_info
        price = info['last_price']
        dates = list(tk.options)
        earn_info = "Keine Daten"
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_info = cal['Earnings Date'][0].strftime('%d.%m.%Y')
        except: pass
        return price, dates, earn_info
    except: return None, [], "Fehler"

def suggest_repair_call(ticker, current_price, dates):
    if not dates or len(dates) < 2: return None, None
    try:
        tk = yf.Ticker(ticker)
        # Wähle Laufzeit in ca. 30 Tagen
        target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
        chain = tk.option_chain(target_date).calls
        T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
        # Suche Call mit Delta ~0.30
        chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(current_price, r['strike'], T, r['impliedVolatility'] or 0.4, 'call'), axis=1)
        valid = chain[chain['delta'] <= 0.35].sort_values('delta', ascending=False)
        if not valid.empty:
            return valid.iloc[0], target_date
    except: pass
    return None, None

# --- UI SETUP ---
st.set_page_config(page_title="CapTrader AI Guard", layout="wide")
st.title("🤖 CapTrader AI Portfolio Guard")

# --- SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit OTM (%)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
min_stock_p = st.sidebar.number_input("Mindestkurs Aktie ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs Aktie ($)", value=600)

# --- SEKTION 1: DEPOT ANALYSE & REPAIR ---
st.header("💼 Aktive Depot-Überwachung")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 51.07},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

d_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, dates, earn = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i % 3]:
            st.markdown(f"### {item['Ticker']}")
            st.metric("Kurs", f"{price:.2f}$", f"{perf:.1f}%")
            st.caption(f"Earnings: {earn}")
            
            if perf < -20:
                st.warning("🚨 Reparatur-Bedarf!")
                rep, d_date = suggest_repair_call(item['Ticker'], price, dates)
                if rep is not None:
                    st.info(f"**KI-Repair:** {d_date} Call @{rep['strike']}$\nPrämie: ~{rep['bid']*100:.0f}$")
            st.write("---")

# --- SEKTION 2: SCANNER ---
st.write("## 🚀 Markt-Scan (Nasdaq & S&P 500)")
if st.button("Jetzt scannen"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "HOOD", "SQ"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i+1)/len(watchlist))
        p, d, e = get_stock_basics(t)
        if p and min_stock_p <= p <= max_stock_p and d:
            try:
                tk = yf.Ticker(t)
                target_date = min(d, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1) / 365
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4, 'put'), axis=1)
                matches = chain[(chain['delta'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)].copy()
                if not matches.empty:
                    best = matches.sort_values('bid', ascending=False).iloc[0]
                    y_pa = (best['bid'] / best['strike']) * (365 / (T*365)) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'Ticker': t, 'Yield p.a.': f"{y_pa:.1f}%", 'Strike': best['strike'], 'Puffer': f"{(abs(best['strike']-p)/p)*100:.1f}%", 'Margin': f"{best['strike']*100:,.0f}$"})
            except: continue
    
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("Keine Treffer. Tipp: Mindestrendite p.a. auf 15-20% senken.")
