import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP & STYLE (Kompatibel mit allen Versionen) ---
st.set_page_config(page_title="CapTrader AI", layout="wide")

# CSS für das Dashboard-Design aus Bild 6, aber ohne Absturzrisiko
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; border: 1px solid #e1e4e8; }
    .main-title { font-size: 2rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- MATHE & DATEN ---
def get_delta(S, K, T, sigma, cp='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1 if cp == 'put' else norm.cdf(d1)

def get_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    up = delta.clip(lower=0).rolling(window=window).mean()
    down = -delta.clip(upper=0).rolling(window=window).mean()
    return 100 - (100 / (1 + up/down))

@st.cache_data(ttl=600)
def get_stock(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        hist = tk.history(period="1mo")
        rsi = get_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        # Fix für Bild 4: Sicherere Earnings-Abfrage
        earn = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, list(tk.options), rsi, earn, tk
    except: return None, [], 50, "", None

# --- SIDEBAR (Fix für Bild 7 & 8) ---
st.sidebar.header("Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield = st.sidebar.number_input("Min. Rendite p.a. (%)", value=15)
# checkbox statt toggle (Fix Bild 7)
sort_rsi = st.sidebar.checkbox("Nach RSI sortieren", value=False)
st.sidebar.markdown("---") # Ersatz für divider (Fix Bild 8)

st.markdown('<p class="main-title">🛡️ CapTrader AI Market Intelligence</p>', unsafe_allow_html=True)

# --- SCANNER ---
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM"]
    results = []
    for t in watchlist:
        p, dates, rsi, earn, tk = get_stock(t)
        if p and dates:
            try:
                d_target = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(d_target).puts
                T = (datetime.strptime(d_target, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                
                # Filterung & Rendite
                matches = chain[chain['delta'].abs() <= (1 - target_prob/100)].copy()
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (1/T) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield:
                        results.append({'T': t, 'Y': best['y_pa'], 'S': best['strike'], 'B': best['bid'], 'D': abs(best['delta']), 'R': rsi})
            except: continue

    if results:
        df = pd.DataFrame(results).sort_values('R' if sort_rsi else 'Y', ascending=False)
        cols = st.columns(3)
        for i, r in enumerate(df.to_dict('records')):
            with cols[i % 3]:
                st.metric(r['T'], f"{r['Y']:.1f}% p.a.", f"Δ {r['D']:.2f}")
                st.write(f"**Strike:** {r['S']}$ | **Bid:** {r['B']:.2f}$")

# --- DEPOT MANAGER (Fix Bild 3 & 6) ---
st.markdown("---")
st.subheader("💼 Smart Depot-Manager")
depot = ["AFRM", "HOOD", "NVDA", "PLTR", "META"]
d_cols = st.columns(3)
for i, t in enumerate(depot):
    p, _, rsi, earn, _ = get_stock(t)
    if p:
        with d_cols[i % 3]:
            # Expander statt Container(border=True) (Fix Bild 3)
            with st.expander(f"{t} Analysis", expanded=True):
                st.write(f"Kurs: **{p:.2f}$** | RSI: **{rsi:.0f}**")
                if rsi > 65: st.success("🎯 Tipp: Call verkaufen")
                elif rsi < 35: st.info("💎 Tipp: Hold (Oversold)")
                if earn: st.caption(f"Earnings: {earn}")

# --- EINZEL-CHECK (DELTA & PRÄMIE GARANTIERT) ---
st.markdown("---")
st.subheader("🔍 Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"])
with c2: t_in = st.text_input("Symbol eingeben", "NVDA").upper()

if t_in:
    p, dates, rsi, earn, tk = get_stock(t_in)
    if p and dates:
        st.write(f"Kurs: {p:.2f}$ | RSI: {rsi:.0f}")
        d_sel = st.selectbox("Laufzeit", dates)
        try:
            chain = tk.option_chain(d_sel).puts if mode == 'put' else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4, mode), axis=1)
            
            for _, row in chain[chain['delta'].abs() < 0.35].sort_values('strike', ascending=(mode=='call')).head(5).iterrows():
                # Delta und Prämie direkt in der Kopfzeile
                with st.expander(f"Strike {row['strike']}$ | Δ {abs(row['delta']):.2f} | Bid {row['bid']:.2f}$"):
                    st.write(f"💰 **Cash-Einnahme:** {row['bid']*100:.0f}$")
                    st.write(f"📉 **Puffer zum Kurs:** {abs(row['strike']-p)/p*100:.1f}%")
        except: st.error("Datenfehler.")
