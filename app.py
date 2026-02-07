import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP & ROBUSTE MATHE ---
st.set_page_config(page_title="Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet Delta mit 45% Fallback-Vola, falls Yahoo 0 liefert."""
    T = max(T, 0.0001)
    # Wenn IV 0 ist, nutzen wir 0.45 als Sicherheitsanker für Delta-Berechnung
    sig = sigma if (sigma and sigma > 0.05) else 0.45
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        val = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        return round(float(val), 2)
    except:
        return -0.35 if option_type == 'put' else 0.35

@st.cache_data(ttl=300)
def get_data(symbol):
    """Holt Kurs und Laufzeiten. Fehlerresistent."""
    try:
        tk = yf.Ticker(symbol)
        return tk.fast_info['last_price'], list(tk.options)
    except:
        return None, []

# --- 2. SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. SCANNER (FIX FÜR LARGEUTF8) ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "AFRM", "SQ", "RIVN", "MSTR"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        p, dates = get_data(t)
        if p and dates:
            try:
                tk = yf.Ticker(t)
                d_target = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(d_target).puts
                days = max((datetime.strptime(d_target, '%Y-%m-%d') - datetime.now()).days, 1)
                # Delta berechnen
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(p, r['strike'], days/365, r['impliedVolatility']), axis=1)
                # Filter
                m = chain[(chain['delta'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)].copy()
                if not m.empty:
                    m['y'] = (m['bid'] / m['strike']) * (365 / days) * 100
                    best = m.sort_values('y', ascending=False).iloc[0]
                    if best['y'] >= min_yield_pa:
                        results.append({"Ticker": str(t), "Rendite": f"{best['y']:.1f}%", "Strike": float(best['strike']), "Kurs": f"{p:.2f}$"})
            except: continue
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else: st.warning("Keine Treffer.")

st.markdown("---")

# --- 4. DEPOT-STATUS (ALLE WERTE) ---
st.subheader("💼 Depot-Status")
depot = ["AFRM", "ELF", "ETSY", "GTLB", "GTM", "HIMS", "HOOD", "JKS", "NVO", "RBRK", "SE", "TTD"]
d_cols = st.columns(4)
for i, t in enumerate(depot):
    p, _ = get_data(t)
    if p:
        with d_cols[i % 4]:
            st.metric(t, f"{p:.2f}$")

st.markdown("---")

# --- 5. EINZEL-CHECK (RESET-FIX) ---
st.subheader("🔍 Experten Einzel-Check")

# Hard-Reset Logik für Ticker-Wechsel
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = 'ELF'

c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True, key="m_radio")
with c2: 
    t_input = st.text_input("Ticker Symbol", value=st.session_state.current_ticker).upper().strip()

# Wenn der Ticker im Feld geändert wird -> Session-State Update & Rerun
if t_input != st.session_state.current_ticker:
    st.session_state.current_ticker = t_input
    st.rerun()

price, dates = get_data(st.session_state.current_ticker)

if price and dates:
    st.write(f"Aktueller Kurs **{st.session_state.current_ticker}**: **{price:.2f}$**")
    # Dynamischer Key verhindert das "Einfrieren" der Selectbox
    d_sel = st.selectbox("Laufzeit wählen", dates, key=f"sb_{st.session_state.current_ticker}")
    
    try:
        tk = yf.Ticker(st.session_state.current_ticker)
        chain = tk.option_chain(d_sel)
        df = chain.puts if mode == "put" else chain.calls
        T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
        
        # Strike-Logik: Nah am Kurs zuerst
        if mode == "put":
            df = df[df['strike'] <= price * 1.03].sort_values('strike', ascending=False)
        else:
            df = df[df['strike'] >= price * 0.97].sort_values('strike', ascending=True)
            
        for _, opt in df.head(8).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'], mode)
            d_abs = abs(delta)
            
            # Ampel
            is_itm = (mode == "put" and opt['strike'] > price) or (mode == "call" and opt['strike'] < price)
            icon = "🔴 ITM" if is_itm else "🟢 OTM" if d_abs < 0.25 else "🟡 RISK"
            
            with st.expander(f"{icon} | Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$ | Delta: {d_abs:.2f}"):
                la, lb = st.columns(2)
                with la:
                    st.write(f"📊 **OTM-Chance:** {(1-d_abs)*100:.1f}%")
                    st.write(f"💰 **Einnahme:** {opt['bid']*100:.0f}$")
                with lb:
                    st.write(f"🎯 **Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                    st.write(f"💼 **Kapital:** {opt['strike']*100:,.0f}$")
    except:
        st.error("Daten konnten nicht geladen werden.")
