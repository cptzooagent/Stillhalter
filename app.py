import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time

# --- 1. SETUP ---
st.set_page_config(page_title="CapTrader AI Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d1) - 1) if option_type == 'put' else float(norm.cdf(d1))
    except: return 0.0

@st.cache_data(ttl=300)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        inf = tk.fast_info
        price = inf['last_price']
        # Wichtig: Hole die Daten mit einem kleinen Timeout/Retry
        for _ in range(3):
            dates = tk.options
            if dates: break
            time.sleep(0.2)
        
        hist = tk.history(period="1mo")
        rsi = 50.0
        if len(hist) >= 14:
            delta = hist['Close'].diff()
            up = delta.clip(lower=0).rolling(window=14).mean()
            down = -delta.clip(upper=0).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + up/down)).iloc[-1]
        return float(price), list(dates), round(float(rsi), 1)
    except:
        return None, [], 50.0

# --- 2. SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. DEPOT ---
st.subheader("💼 Depot-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "NVO", "Einstand": 97.0},
    {"Ticker": "RBRK", "Einstand": 70.0}, {"Ticker": "SE", "Einstand": 170.0},
    {"Ticker": "TTD", "Einstand": 102.0}
]

cols = st.columns(3)
for i, d in enumerate(depot_data):
    p, _, _ = get_stock_basics(d['Ticker'])
    if p:
        perf = (p/d['Einstand']-1)*100
        with cols[i%3]:
            st.metric(d['Ticker'], f"{p:.2f} $", f"{perf:.1f} %", delta_color="normal" if perf > -15 else "inverse")
            if perf < -15 and st.button(f"Fix {d['Ticker']}", key=f"f_{d['Ticker']}"):
                st.session_state['active_ticker'] = d['Ticker']

st.markdown("---")

# --- 4. EINZEL-CHECK (ROBUST) ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: opt_type = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker", value=st.session_state.get('active_ticker', 'ELF')).upper()

if t_input:
    price, dates, rsi = get_stock_basics(t_input)
    if price and dates:
        st.write(f"**Kurs:** {price:.2f}$ | **RSI:** {rsi}")
        d_sel = st.selectbox("Laufzeit", dates)
        
        # Lade-Logik mit Fehlerbehandlung
        try:
            tk = yf.Ticker(t_input)
            chain = tk.option_chain(d_sel)
            df = chain.puts if opt_type == 'put' else chain.calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # Filter: Zeige Puts unter Kurs, Calls über Kurs + Puffer für ITM/OTM
            if opt_type == 'put':
                opts = df[df['strike'] <= price * 1.1].sort_values('strike', ascending=False)
            else:
                opts = df[df['strike'] >= price * 0.9].sort_values('strike', ascending=True)
            
            if opts.empty:
                st.info("Keine handelbaren Strikes für diese Auswahl gefunden (Bid = 0).")
            else:
                for _, row in opts.head(8).iterrows():
                    delta = calculate_bsm_delta(price, row['strike'], T, row['impliedVolatility'] or 0.3, opt_type)
                    otm = (1 - abs(delta)) * 100
                    status = "🟢" if otm > 80 else "🟡" if otm > 60 else "🔴"
                    
                    with st.expander(f"{status} Strike {row['strike']} | Prämie: {row['bid']}$ | OTM: {otm:.1f}%"):
                        st.write(f"Delta: {delta:.2f} | Kapital: {row['strike']*100:.0f}$ | Rendite: {(row['bid']/row['strike'])*(1/T)*100:.1f}% p.a.")
        except Exception as e:
            st.error(f"Daten-Fehler: Bitte wähle eine andere Laufzeit oder versuche es in 10 Sek. erneut.")
