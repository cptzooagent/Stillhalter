import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# --- 1. MATHE-FUNKTIONEN ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-ABRUF ---
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
        except: 
            earn_str = ""
        
        return price, list(tk.options), earn_str, rsi_val
    except:
        return None, [], "", 50

# --- UI: SIDEBAR ---
st.sidebar.header("Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 99, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=15)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: MARKT-SCAN (Fix für Laufzeit-Anzeige) ---
if st.button("🚀 Markt-Scan starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "AMD", "NFLX", "COIN"]
    results = []
    
    for t in watchlist:
        price, dates, earn, rsi = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                # Option wählen (ca. 30 Tage DTE)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                T = max(1/365, days_to_expiry / 365)
                
                chain = tk.option_chain(target_date).puts
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4, option_type='put'), axis=1)
                
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                if not safe_opts.empty:
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (1/T) * 100
                    best = safe_opts.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield_pa:
                        results.append({
                            'T': t, 'Y': best['y_pa'], 'S': best['strike'], 
                            'B': best['bid'], 'D': abs(best['delta_val']), 
                            'R': rsi, 'E': earn, 'Days': days_to_expiry, 'Exp': target_date
                        })
            except:
                continue

    if results:
        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i % 3]:
                st.markdown(f"### {r['T']}")
                st.metric("Rendite p.a.", f"{r['Y']:.1f}%", f"Δ {r['D']:.2f}")
                st.write(f"📅 **Ablauf:** {r['Exp']} ({r['Days']} Tage)")
                st.write(f"💰 **Cash-Prämie:** {r['B']*100:.0f}$ | Strike: {r['S']}$")
                if r['E']: st.warning(f"Earnings: {r['E']}")
    else:
        st.info("Keine Treffer unter den aktuellen Einstellungen.")

# --- SEKTION 2: DEPOT-MANAGER ---
st.write("---")
st.subheader("💼 Smart Depot-Manager")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "HOOD", "Einstand": 82.82}, {"Ticker": "NVO", "Einstand": 97.0}
]
p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            st.write(f"**{item['Ticker']}** ({diff:.1f}%)")
            st.caption(f"Kurs: {price:.2f}$ | RSI: {rsi:.0f}")

# --- SEKTION 3: EINZEL-CHECK (Fix für Bild 26 & Delta) ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")

c_type, c_tick = st.columns([1, 3])
opt_type = c_type.radio("Typ", ["put", "call"], horizontal=True)
t_in = c_tick.text_input("Symbol prüfen", "HOOD").upper()

if t_in:
    price, dates, earn, rsi = get_stock_data_full(t_in)
    if price and dates:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}**")
        expiry = st.selectbox("Laufzeit wählen", dates)
        
        tk = yf.Ticker(t_in)
        chain = tk.option_chain(expiry).calls if opt_type == "call" else tk.option_chain(expiry).puts
        T = max(1/365, (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365)
        
        # Delta neu berechnen
        chain['delta_calc'] = chain.apply(
            lambda o: calculate_bsm_delta(price, o['strike'], T, o['impliedVolatility'] or 0.4, option_type=opt_type), axis=1
        )
        
        # Sortierung (Call = aufsteigend, Put = absteigend)
        sort_asc = (opt_type == "call")
        display_chain = chain.sort_values('strike', ascending=sort_asc).head(8)
        
        for _, opt in display_chain.iterrows():
            d_abs = abs(opt['delta_calc'])
            risk_color = "🟢" if d_abs < 0.15 else "🟡" if d_abs < 0.25 else "🔴"
            
            with st.expander(f"{risk_color} Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f}"):
                c1, c2 = st.columns(2)
                # Fix für Bild 26: f-strings sauber geschlossen
                c1.write(f"💰 Prämie: {opt['bid']*100:.0f}$")
                c1.write(f"📉 Puffer: {abs(opt['strike']-price)/price*100:.1f}%")
                c2.write(f"🌊 IV: {int((opt['impliedVolatility'] or 0)*100)}%")
                c2.write(f"🎯 OTM-Prob: {int((1-d_abs)*100)}%")
