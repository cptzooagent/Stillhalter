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
        if option_type == 'call':
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1)
    except: return 0.0

@st.cache_data(ttl=600)
def get_stock_basics(symbol):
    """Holt Preis, Optionen und RSI mit Retry-Logik gegen Datenfehler."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        
        # Retry-Logik für stabilere Daten (behebt Fehler aus Bild 13.png)
        dates = []
        for _ in range(3):
            dates = tk.options
            if dates: break
            time.sleep(0.3)
            
        hist = tk.history(period="3mo")
        rsi = 50.0
        if len(hist) >= 14:
            delta = hist['Close'].diff()
            up = delta.clip(lower=0).rolling(window=14).mean()
            down = -delta.clip(upper=0).rolling(window=14).mean()
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
st.subheader("🚀 Markt-Chancen Scanner")
if st.button("Markt-Scan starten"):
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
    if results:
        st.table(pd.DataFrame(results))
    else: st.warning("Keine Treffer im Scan. Versuche die Filter anzupassen.")

st.markdown("---")

# --- 4. DEPOT-ÜBERWACHUNG ---
st.subheader("💼 Depot-Status")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "ELF", "Einstand": 109.0},
    {"Ticker": "ETSY", "Einstand": 67.0}, {"Ticker": "GTLB", "Einstand": 41.0},
    {"Ticker": "HOOD", "Einstand": 120.0}, {"Ticker": "NVO", "Einstand": 97.0},
    {"Ticker": "RBRK", "Einstand": 70.0}, {"Ticker": "SE", "Einstand": 170.0},
    {"Ticker": "TTD", "Einstand": 102.0}
]

d_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, _, _ = get_stock_basics(item['Ticker'])
    if price:
        perf = (price / item['Einstand'] - 1) * 100
        with d_cols[i % 3]:
            if perf < -15:
                st.error(f"🚨 **{item['Ticker']}**: {price:.2f}$ ({perf:.1f}%)")
                if st.button(f"Reparatur {item['Ticker']}", key=f"rep_{item['Ticker']}"):
                    st.session_state['active_ticker'] = item['Ticker']
            else:
                st.success(f"✅ **{item['Ticker']}**: {price:.2f}$ ({perf:.1f}%)")

st.markdown("---")

# --- 5. EXPERTEN EINZEL-CHECK ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: opt_type = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker Symbol", value=st.session_state.get('active_ticker', 'ELF')).upper()

if t_input:
    price, dates, rsi, trend = get_stock_basics(t_input)
    if price and dates:
        m1, m2, m3 = st.columns(3)
        m1.metric("Kurs", f"{price:.2f}$")
        m2.metric("RSI", rsi)
        m3.metric("Trend", trend)
        
        d_sel = st.selectbox("Laufzeit wählen", dates)
        try:
            tk = yf.Ticker(t_input)
            # Umschalten zwischen Puts und Calls
            df = tk.option_chain(d_sel).puts if opt_type == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # Strike-Sortierung (Wichtig für ELF!)
            if opt_type == "put":
                df = df[df['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                df = df[df['strike'] >= price * 0.95].sort_values('strike', ascending=True)
            
            for _, opt in df.head(10).iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, opt_type)
                prob_otm = (1 - abs(delta)) * 100
                ampel = "🟢" if abs(delta) <= 0.20 else "🟡" if abs(delta) <= 0.40 else "🔴"
                
                with st.expander(f"{ampel} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | OTM: {prob_otm:.1f}%"):
                    st.write(f"💰 Einnahme: **{opt['bid']*100:.0f}$** | Delta: **{delta:.2f}**")
        except:
            st.error("Optionskette konnte nicht geladen werden.")
