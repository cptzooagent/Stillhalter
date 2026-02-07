import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Guard Pro", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0: return 0.5
    sigma = max(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)

# --- 2. DATEN-FUNKTIONEN (OHNE CACHING-FEHLER) ---
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        # Wir holen nur Rohdaten ab
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        return float(price), dates
    except:
        return None, []

def get_rsi_simple(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="1mo")
        if len(hist) < 14: return 50.0
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        return round(float(rsi), 1)
    except: return 50.0

# --- 3. SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

# --- 4. DEPOT-ÜBERWACHUNG ---
st.title("🛡️ CapTrader AI Market Guard Pro")

depot_list = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "GTLB", "Einstand": 41.0}, {"Ticker": "HOOD", "Einstand": 120.0},
    {"Ticker": "TTD", "Einstand": 102.0}, {"Ticker": "SE", "Einstand": 170.0}
]

st.subheader("💼 Depot-Status")
d_cols = st.columns(3)
for i, item in enumerate(depot_list):
    price, _ = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i % 3]:
            if perf < -15:
                st.error(f"🚨 **{item['Ticker']}**: {price:.2f}$ ({perf:.1f}%)")
                if st.button(f"Reparatur {item['Ticker']}", key=f"rep_{item['Ticker']}"):
                    st.session_state['active_ticker'] = item['Ticker']
            else:
                st.success(f"✅ **{item['Ticker']}**: {price:.2f}$ ({perf:.1f}%)")

st.write("---")

# --- 5. EINZEL-CHECK (ROBUST & KORREKT) ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: opt_type = st.radio("Typ wählen", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker Symbol", value=st.session_state.get('active_ticker', 'ELF')).upper()

if t_input:
    price, dates = get_stock_basics(t_input)
    if price and dates:
        rsi = get_rsi_simple(t_input)
        m1, m2 = st.columns(2)
        m1.metric("Aktueller Kurs", f"{price:.2f}$")
        m2.metric("RSI (14d)", rsi)
        
        d_sel = st.selectbox("Laufzeit wählen", dates)
        
        try:
            tk = yf.Ticker(t_input)
            chain = tk.option_chain(d_sel)
            df = chain.puts if opt_type == "put" else chain.calls
            
            # Laufzeit in Jahren für Delta
            expiry_dt = datetime.strptime(d_sel, '%Y-%m-%d')
            T = (expiry_dt - datetime.now()).days / 365
            
            # Filter für sinnvolle Strikes (OTM Fokus)
            if opt_type == "put":
                # Puts: Strikes von "At-the-Money" nach unten
                df_filtered = df[df['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                # Calls: Strikes von "At-the-Money" nach oben
                df_filtered = df[df['strike'] >= price * 0.95].sort_values('strike', ascending=True)

            if df_filtered.empty:
                st.warning("Keine handelbaren Optionen für diesen Bereich gefunden.")
            else:
                for _, opt in df_filtered.head(10).iterrows():
                    delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, opt_type)
                    prob_otm = (1 - abs(delta)) * 100
                    
                    # Ampel-System
                    color = "🟢" if abs(delta) < 0.2 else "🟡" if abs(delta) < 0.4 else "🔴"
                    
                    with st.expander(f"{color} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | OTM: {prob_otm:.1f}%"):
                        col_a, col_b = st.columns(2)
                        col_a.write(f"💰 **Einnahme: {opt['bid']*100:.0f}$**")
                        col_b.write(f"📉 **Delta: {delta:.2f}**")
                        if abs(delta) > 0.5:
                            st.info("ℹ️ Diese Option ist 'In-the-Money'.")
        except Exception as e:
            st.error(f"Fehler beim Laden der Optionskette: {e}")
    else:
        st.error("Ticker nicht gefunden oder keine Daten verfügbar.")
