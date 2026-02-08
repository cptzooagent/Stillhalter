import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. STABILE KONFIGURATION ---
st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")

# Custom CSS für die Karten-Optik aus Bild 6
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; border: 1px solid #e1e4e8; }
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .risk-note { font-size: 0.9rem; color: #64748b; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIK-KERN ---
def get_delta(S, K, T, sigma, cp='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (0.04 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1 if cp == 'put' else norm.cdf(d1)

def get_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    up = delta.clip(lower=0).rolling(window=window).mean()
    down = -delta.clip(upper=0).rolling(window=window).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=600)
def get_stock_info(symbol):
    try:
        tk = yf.Ticker(symbol)
        p = tk.fast_info['last_price']
        h = tk.history(period="1mo")
        rsi_val = get_rsi(h['Close']).iloc[-1] if not h.empty else 50
        earn = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return p, list(tk.options), rsi_val, earn, tk
    except: return None, [], 50, "", None

# --- 3. SIDEBAR (Fix Bild 7 & 8) ---
st.sidebar.header("⚙️ Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield = st.sidebar.number_input("Min. Rendite p.a. (%)", value=15)
sort_rsi = st.sidebar.checkbox("Nach RSI sortieren", value=True)
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ RSI immer mit Chart (Support/Resistance) abgleichen!")

st.markdown('<p class="main-title">🛡️ CapTrader AI Market Intelligence</p>', unsafe_allow_html=True)

# --- 4. SCANNER (STABIL & DETAILREICH) ---
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        p, dates, rsi, earn, tk = get_stock_info(t)
        if p and dates:
            try:
                # Target: ca. 30 Tage Laufzeit
                d_target = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(d_target).puts
                T = (datetime.strptime(d_target, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                
                matches = chain[chain['delta'].abs() <= (1 - target_prob/100)].copy()
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (1/T) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield:
                        results.append({'T': t, 'Y': best['y_pa'], 'S': best['strike'], 'B': best['bid'], 'D': abs(best['delta']), 'R': rsi, 'E': earn})
            except: continue

    if results:
        df = pd.DataFrame(results).sort_values('R' if sort_rsi else 'Y', ascending=(not sort_rsi))
        cols = st.columns(3)
        for i, r in enumerate(df.to_dict('records')):
            with cols[i % 3]:
                st.metric(r['T'], f"{r['Y']:.1f}% p.a.", f"Δ {r['D']:.2f}")
                with st.expander(f"Details: Strike {r['S']}$", expanded=False):
                    st.write(f"💵 **Prämie:** {r['B']:.2f}$ ({r['B']*100:.0f}$ Cash)")
                    st.write(f"📊 **RSI:** {r['R']:.0f}")
                    if r['E']: st.error(f"📅 Earnings: {r['E']}")
    else:
        st.warning("Keine Treffer mit diesen Filtern.")

# --- 5. DEPOT-MANAGER (Fix Bild 3 & 6) ---
st.markdown("---")
st.subheader("💼 Smart Depot-Manager")
depot = ["AFRM", "HOOD", "NVDA", "PLTR", "META", "TSLA"]
d_cols = st.columns(3)
for i, t in enumerate(depot):
    p, _, rsi, earn, _ = get_stock_info(t)
    if p:
        with d_cols[i % 3]:
            # Expander ist die sicherste Lösung für deine Streamlit-Version (Bild 3)
            with st.expander(f"{t} Analyse", expanded=True):
                m1, m2 = st.columns(2)
                m1.metric("Kurs", f"{p:.1f}$")
                m2.metric("RSI", f"{rsi:.0f}")
                if rsi < 30: st.info("💎 Oversold - Hold")
                elif rsi > 70: st.success("🎯 Overbought - Sell Call?")
                if earn: st.caption(f"Earnings am {earn}")

# --- 6. EINZEL-CHECK (DELTA & PRÄMIE IM TITEL) ---
st.markdown("---")
st.subheader("🔍 Deep-Dive Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Optionstyp", ["put", "call"])
with c2: t_in = st.text_input("Ticker-Symbol", "NVDA").upper()

if t_in:
    p, dates, rsi, earn, tk = get_stock_info(t_in)
    if p and dates:
        st.write(f"Aktueller Kurs: **{p:.2f}$** | RSI: **{rsi:.0f}**")
        d_sel = st.selectbox("Optionen-Laufzeit wählen", dates)
        try:
            chain = tk.option_chain(d_sel).puts if mode == 'put' else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta'] = chain.apply(lambda r: get_delta(p, r['strike'], T, r['impliedVolatility'] or 0.4, mode), axis=1)
            
            # Anzeige der Top 5 Strikes mit Delta und Prämie direkt im Titel
            for _, row in chain[chain['delta'].abs() < 0.4].sort_values('strike', ascending=(mode=='call')).head(5).iterrows():
                d_val = abs(row['delta'])
                with st.expander(f"Strike {row['strike']:.1f}$ | Δ {d_val:.2f} | Bid: {row['bid']:.2f}$"):
                    a, b = st.columns(2)
                    a.write(f"💰 **Einnahme:** {row['bid']*100:.0f}$")
                    b.write(f"📉 **Puffer:** {abs(row['strike']-p)/p*100:.1f}%")
        except: st.error("Konnte Optionsdaten nicht laden.")
