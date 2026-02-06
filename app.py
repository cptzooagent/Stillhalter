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

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_date = None
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0]
        except: pass
        return price, dates, earn_date
    except: return None, [], None

def suggest_repair_call(ticker, current_price, dates):
    if not dates: return None
    try:
        tk = yf.Ticker(ticker)
        target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
        chain = tk.option_chain(target_date).calls
        T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
        # Suche Call mit Delta ~0.30 für Reparatur
        chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(current_price, r['strike'], T, r['impliedVolatility'] or 0.4, 'call'), axis=1)
        best_repair = chain[chain['delta'] <= 0.35].sort_values('delta', ascending=False).iloc[0]
        return best_repair, target_date
    except: return None, None

# --- UI SETUP ---
st.set_page_config(page_title="CapTrader AI Guard", layout="wide")
st.title("🤖 CapTrader AI Portfolio Guard & Repair")

# --- SIDEBAR ---
st.sidebar.header("🛡️ Scanner-Filter")
target_prob = st.sidebar.slider("Sicherheit OTM (%)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
min_stock_p = st.sidebar.number_input("Mindestkurs Aktie ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs Aktie ($)", value=600)

# --- SEKTION 1: DEPOT ANALYSE ---
st.subheader("💼 Aktive Depot-Überwachung")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

d_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, dates, earn = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i % 3]:
            # Kompatibler Container-Ersatz
            st.markdown(f"### {item['Ticker']}")
            st.metric("Performance", f"{price:.2f}$", f"{perf:.1f}%")
            
            if perf < -20:
                st.error(f"🚨 Repair-Bedarf!")
                rep, d_date = suggest_repair_call(item['Ticker'], price, dates)
                if rep is not None:
                    st.info(f"**KI-Repair:** Verkauf {d_date} Call @{rep['strike']}$ \nPrämie: ~{rep['bid']*100:.0f}$")
            
            if earn:
                days = (earn.date() - datetime.now().date()).days
                if 0 <= days <= 5: st.warning(f"📅 Earnings in {days} Tagen!")
            st.write("---")

# --- SEKTION 2: SCANNER ---
st.divider()
st.subheader("🚀 Markt-Chancen (S&P 500 & Nasdaq)")
if st.button("Jetzt Markt scannen"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "HOOD", "SQ", "PYPL"]
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
                        results.append({'Ticker': t, 'Rendite p.a.': f"{y_pa:.1f}%", 'Strike': best['strike'], 'Puffer': f"{(abs(best['strike']-p)/p)*100:.1f}%", 'Kapital': f"{best['strike']*100:,.0f}$"})
            except: continue
    
    if results: st.table(pd.DataFrame(results))
    else: st.warning("Keine Treffer mit diesen Filtern.")
