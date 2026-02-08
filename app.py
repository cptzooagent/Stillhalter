import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. MATHE: DELTA & RSI ---
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

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        return price, list(tk.options), earn_str, rsi_val
    except:
        return None, [], "", 50

# --- UI: SIDEBAR ---
st.sidebar.header("Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 99, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: KOMBI-SCAN (Wie Bild 15) ---
if st.button("🚀 Markt-Scan starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "NFLX", "COIN"]
    results = []
    
    for t in watchlist:
        price, dates, earn, rsi = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max(1/365, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365)
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (1/T) * 100
                    best = safe_opts.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield_pa:
                        results.append({'T': t, 'Y': best['y_pa'], 'S': best['strike'], 'B': best['bid'], 'D': abs(best['delta_val']), 'R': rsi, 'E': earn})
            except: continue

    if results:
        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.markdown(f"### {r['T']}")
                st.metric("Rendite p.a.", f"{r['Y']:.1f}%", f"↑ Δ {r['D']:.2f}")
                st.write(f"💰 **Cash-Prämie: {r['B']*100:.0f}$**")
                st.write(f"🎯 Strike: {r['S']}$ | RSI: {r['R']:.0f}")
                if r['E']: st.warning(f"📅 Earnings: {r['E']}")
    else: st.info("Keine Treffer unter den aktuellen Einstellungen.")

# --- SEKTION 2: DEPOT-MANAGER (Wie Bild 6) ---
st.write("---")
st.subheader("💼 Smart Depot-Manager")

depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "GTM", "Einstand": 17.0}, {"Ticker": "HIMS", "Einstand": 37.0},
    {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "JKS", "Einstand": 50.0},
    {"Ticker": "NVO", "Einstand": 97.0}, {"Ticker": "RBRK", "Einstand": 70.0},
    {"Ticker": "SE", "Einstand": 170.0}, {"Ticker": "TTD", "Einstand": 102.0}
]

p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            with st.expander(f"{item['Ticker']} ({diff:.1f}%)", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Kurs", f"{price:.2f}$")
                c2.metric("RSI", f"{rsi:.0f}")
                if rsi < 30: st.info("💎 Oversold - Hold")
                elif rsi > 70: st.success("🎯 Overbought - Sell Call?")
                if earn: st.caption(f"📅 Earnings: {earn}")

# --- SEKTION 3: EINZEL-CHECK (Inklusive Delta-Fix) ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")
t_in = st.text_input("Symbol eingeben", "hood").upper()

if t_in:
    price, dates, earn, rsi = get_stock_data_full(t_in)
    if price:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        expiry = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(expiry).puts
        T = max(1/365, (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365)
        
        # Delta-Berechnung für die Liste
        chain['delta_calc'] = chain.apply(lambda o: calculate_bsm_delta(price, o['strike'], T, o['impliedVolatility'] or 0.4), axis=1)
        
        for _, opt in chain[chain['delta_calc'].abs() < 0.45].sort_values('strike', ascending=False).head(8).iterrows():
            d_abs = abs(opt['delta_calc'])
            
            # Risiko-Ampel
            if d_abs < 0.15: risk_icon = "🟢 (Sicher)"
            elif d_abs < 0.25: risk_icon = "🟡 (Moderat)"
            else: risk_icon = "🔴 (Aggressiv)"
            
            # DELTA wieder im Titel und in den Details integriert
            with st.expander(f"{risk_icon} Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"💰 **Cash-Einnahme:** {opt['bid']*100:.0f}$")
                    st.write(f"📉 **Delta:** {d_abs:.2f}")
                with col_b:
                    st.write(f"🎯 **Puffer zum Kurs:** {abs(opt['strike']-price)/price*100:.1f}%")
                    st.write(f"🌊 **Implizite Vola:** {int((opt['impliedVolatility'] or 0)*100)}%")
