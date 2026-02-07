import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP & KONFIGURATION ---
st.set_page_config(page_title="CapTrader AI Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet das Delta einer Option nach Black-Scholes."""
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)

# --- 2. DATEN-FUNKTIONEN (ROBUST & SERIALISIERBAR) ---
def get_stock_basics(symbol):
    """Holt Preis und verfügbare Verfallstermine."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        return float(price), dates
    except:
        return None, []

def get_market_metrics(symbol):
    """Berechnet RSI und Trend-Status."""
    try:
        hist = yf.Ticker(symbol).history(period="3mo")
        if len(hist) < 20: return 50.0, "Unklar"
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        sma50 = hist['Close'].tail(50).mean()
        trend = "Stabil" if hist['Close'].iloc[-1] > (sma50 * 0.95) else "Schwach"
        return round(float(rsi), 1), trend
    except:
        return 50.0, "Fehler"

# --- 3. SIDEBAR (STRATEGIE-FILTER) ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
rsi_min = st.sidebar.slider("Minimum RSI", 20, 45, 30)

# --- 4. SCANNER (NEUE CHANCEN) ---
st.title("🛡️ CapTrader AI Market Guard Pro")

if st.button("🚀 High-Safety Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "MSTR", "UBER", "DIS", "PYPL"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates = get_stock_basics(t)
        if price and dates:
            rsi, trend = get_market_metrics(t)
            if rsi >= rsi_min:
                try:
                    tk = yf.Ticker(t)
                    # Wähle Termin in ca. 30 Tagen
                    target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                    chain = tk.option_chain(target_date).puts
                    T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                    
                    # Delta für alle Puts berechnen
                    chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4, 'put'), axis=1)
                    
                    # Filter: Delta entspricht Sicherheit
                    matches = chain[(chain['delta_val'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)]
                    if not matches.empty:
                        best = matches.sort_values('bid', ascending=False).iloc[0]
                        y_pa = (best['bid'] / best['strike']) * (1/T if T > 0 else 1) * 100
                        if y_pa >= min_yield_pa:
                            results.append({'Ticker': t, 'Rendite p.a.': f"{y_pa:.1f}%", 'Strike': best['strike'], 'RSI': rsi, 'Kurs': f"{price:.2f}$"})
                except: continue
    if results:
        st.table(pd.DataFrame(results))
    else: st.warning("Keine Treffer im Scan gefunden.")

st.write("---")

# --- 5. DEPOT-ÜBERWACHUNG (DEINE POSITIONEN) ---
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

# --- 6. EXPERTEN EINZEL-CHECK (MANUELLES ROLLEN) ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: opt_type = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker Symbol", value=st.session_state.get('active_ticker', 'ELF')).upper()

if t_input:
    price, dates = get_stock_basics(t_input)
    if price and dates:
        rsi, trend = get_market_metrics(t_input)
        m1, m2, m3 = st.columns(3)
        m1.metric("Kurs", f"{price:.2f}$")
        m2.metric("RSI", rsi)
        m3.metric("Trend", trend)
        
        d_sel = st.selectbox("Laufzeit wählen", dates)
        try:
            tk = yf.Ticker(t_input)
            chain = tk.option_chain(d_sel).puts if opt_type == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # STRIKE-FILTER LOGIK
            if opt_type == "put":
                # Zeige Puts vom Kurs abwärts (Sicherheit)
                df = chain[chain['strike'] <= price * 1.1].sort_values('strike', ascending=False)
            else:
                # Zeige Calls vom Kurs aufwärts (Covered Call)
                df = chain[chain['strike'] >= price * 0.9].sort_values('strike', ascending=True)
            
            df = df[df['bid'] > 0].head(10)
            
            for _, opt in df.iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, opt_type)
                prob_otm = (1 - abs(delta)) * 100
                ampel = "🟢" if abs(delta) <= 0.20 else "🟡" if abs(delta) <= 0.40 else "🔴"
                
                with st.expander(f"{ampel} Strike {opt['strike']:.1f}$ | Prämie: {opt['bid']:.2f}$ | OTM: {prob_otm:.1f}%"):
                    col_a, col_b = st.columns(2)
                    col_a.write(f"💰 Cash-Einnahme: **{opt['bid']*100:.0f}$**")
                    col_a.write(f"🛡️ Puffer: **{(abs(opt['strike']-price)/price)*100:.1f}%**")
                    col_b.write(f"📉 Delta: **{delta:.2f}**")
                    col_b.write(f"💼 Kapital: **{opt['strike']*100:,.0f}$**")
        except:
            st.error("Optionskette konnte nicht geladen werden.")
