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
    """Präzise Delta-Berechnung nach Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    try:
        # Standard Black-Scholes Formel für d1
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return float(norm.cdf(d1))
        else: # Put Delta ist immer negativ
            return float(norm.cdf(d1) - 1)
    except Exception:
        return 0.0

@st.cache_data(ttl=600)
def get_stock_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        inf = tk.fast_info
        price = inf['last_price']
        dates = []
        for _ in range(3):
            dates = tk.options
            if dates: break
            time.sleep(0.3)
        hist = tk.history(period="3mo")
        rsi = 50.0
        if len(hist) >= 14:
            delta_p = hist['Close'].diff()
            up = delta_p.clip(lower=0).rolling(window=14).mean()
            down = -delta_p.clip(upper=0).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + up/down)).iloc[-1]
        trend = "Stabil" if price > hist['Close'].tail(50).mean() * 0.95 else "Schwach"
        return float(price), list(dates), round(float(rsi), 1), trend
    except:
        return None, [], 50.0, "Fehler"

# --- 2. SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
rsi_min = st.sidebar.slider("Minimum RSI", 20, 45, 30)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "MSTR", "UBER", "DIS", "PYPL"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, rsi, trend = get_stock_basics(t)
        if price and dates and rsi >= rsi_min:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4, 'put'), axis=1)
                matches = chain[(chain['delta_val'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)]
                if not matches.empty:
                    best = matches.sort_values('bid', ascending=False).iloc[0]
                    y_pa = (best['bid'] / best['strike']) * (1/T if T > 0 else 1) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'Ticker': t, 'Rendite p.a.': f"{y_pa:.1f}%", 'Strike': best['strike'], 'RSI': rsi, 'Kurs': f"{price:.2f}$"})
            except: continue
    if results: st.table(pd.DataFrame(results))
    else: st.warning("Keine Treffer im Scan.")

st.markdown("---")

# --- 4. DEPOT ---
st.subheader("💼 Depot-Status")
depot_data = [{"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0}, {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "SE", "Einstand": 170.0}]
d_cols = st.columns(len(depot_data))
for i, item in enumerate(depot_data):
    price, _, _, _ = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i]:
            if perf < -15: st.error(f"🚨 {item['Ticker']}: {perf:.1f}%")
            else: st.success(f"✅ {item['Ticker']}: {perf:.1f}%")

st.markdown("---")

# --- 5. EINZEL-CHECK (KORRIGIERT) ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: opt_type = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker Symbol", value="ELF").upper()

if t_input:
    price, dates, rsi, trend = get_stock_basics(t_input)
    if price and dates:
        st.write(f"**Kurs:** {price:.2f}$ | **RSI:** {rsi}")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        try:
            tk = yf.Ticker(t_input)
            chain = tk.option_chain(d_sel).puts if opt_type == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # Sortierung: Puts abwärts vom Kurs, Calls aufwärts
            if opt_type == "put":
                df = chain[chain['strike'] <= price * 1.1].sort_values('strike', ascending=False)
            else:
                df = chain[chain['strike'] >= price * 0.9].sort_values('strike', ascending=True)
            
            for _, opt in df.head(10).iterrows():
                # Berechnung des Deltas mit realer Volatilität
                iv = opt['impliedVolatility'] if opt['impliedVolatility'] > 0 else 0.4
                delta = calculate_bsm_delta(price, opt['strike'], T, iv, opt_type)
                
                # OTM Wahrscheinlichkeit basierend auf Delta
                prob_otm = (1 - abs(delta)) * 100
                
                # Dynamische Ampel-Logik
                if opt_type == "put":
                    is_itm = opt['strike'] > price
                else:
                    is_itm = opt['strike'] < price
                
                if is_itm:
                    ampel = "🔴 (ITM)"
                elif abs(delta) > 0.35:
                    ampel = "🟡"
                else:
                    ampel = "🟢"
                
                with st.expander(f"{ampel} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | OTM: {prob_otm:.1f}%"):
                    st.write(f"**Delta:** {delta:.2f} | **Einnahme:** {opt['bid']*100:.0f}$ | **Vola (IV):** {iv:.1%}")
        except: st.error("Fehler beim Laden.")
