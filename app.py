import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP ---
st.set_page_config(page_title="CapTrader Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Präzise Delta-Logik mit Notfall-Anker bei fehlender IV."""
    T = max(T, 0.0001)
    sig = sigma if (sigma and sigma > 0.05) else 0.45
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        return round(float(norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1), 2)
    except:
        return -0.30 if option_type == 'put' else 0.30

@st.cache_data(ttl=300) # Kurzer Cache für schnellere Wechsel
def get_basics(symbol):
    try:
        tk = yf.Ticker(symbol)
        return tk.fast_info['last_price'], list(tk.options)
    except:
        return None, []

# --- 2. SIDEBAR ---
st.sidebar.header("🛡️ Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. SCANNER (STABILISIERT) ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "AFRM", "SQ", "RIVN", "MSTR", "UBER"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        p, dates = get_basics(t)
        if p and dates:
            try:
                tk = yf.Ticker(t)
                d_target = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(d_target).puts
                days = max((datetime.strptime(d_target, '%Y-%m-%d') - datetime.now()).days, 1)
                chain['delta'] = chain.apply(lambda r: calculate_bsm_delta(p, r['strike'], days/365, r['impliedVolatility']), axis=1)
                matches = chain[(chain['delta'].abs() <= (100-target_prob)/100) & (chain['bid'] > 0)].copy()
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (365 / days) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield_pa:
                        results.append({"Ticker": t, "Rendite": f"{best['y_pa']:.1f}%", "Strike": best['strike'], "Kurs": f"{p:.2f}$"})
            except: continue
    if results: st.dataframe(pd.DataFrame(results), use_container_width=True)
    else: st.warning("Keine Treffer.")

st.markdown("---")

# --- 4. EINZEL-CHECK (REAKTIONSSCHNELL) ---
st.subheader("🔍 Einzel-Check")
# Wir nutzen SessionState, um den Ticker stabil zu halten
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'ELF'

col1, col2 = st.columns([1, 2])
with col1:
    mode = st.radio("Typ", ["put", "call"], horizontal=True, key="mode_sel")
with col2:
    new_ticker = st.text_input("Ticker Symbol", value=st.session_state.ticker).upper().strip()

# Wenn ein neuer Ticker eingegeben wurde, Cache-Reset erzwingen
if new_ticker != st.session_state.ticker:
    st.session_state.ticker = new_ticker
    st.rerun()

price, dates = get_basics(st.session_state.ticker)

if price and dates:
    st.write(f"Kurs **{st.session_state.ticker}**: **{price:.2f}$**")
    # Dynamischer Key für die Selectbox verhindert das "Einfrieren"
    d_sel = st.selectbox("Laufzeit", dates, key=f"date_{st.session_state.ticker}")
    
    try:
        tk = yf.Ticker(st.session_state.ticker)
        chain = tk.option_chain(d_sel)
        df = chain.puts if mode == "put" else chain.calls
        T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
        
        # Strike-Filter & Sortierung (Nah am Geld zuerst)
        if mode == "put":
            df = df[df['strike'] <= price * 1.03].sort_values('strike', ascending=False)
        else:
            df = df[df['strike'] >= price * 0.97].sort_values('strike', ascending=True)
        
        for _, opt in df.head(8).iterrows():
            delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'], mode)
            d_abs = abs(delta)
            is_itm = (mode == "put" and opt['strike'] > price) or (mode == "call" and opt['strike'] < price)
            
            icon = "🔴 ITM" if is_itm else "🟢 OTM" if d_abs < 0.25 else "🟡 RISK"
            
            with st.expander(f"{icon} | Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$ | Delta: {d_abs:.2f}"):
                ca, cb = st.columns(2)
                with ca:
                    st.write(f"📊 **OTM:** {(1-d_abs)*100:.1f}%")
                    st.write(f"💰 **Einnahme:** {opt['bid']*100:.0f}$")
                with cb:
                    st.write(f"🎯 **Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                    st.write(f"💼 **Kapital:** {opt['strike']*100:,.0f}$")
    except:
        st.error(f"Keine Optionsdaten für {st.session_state.ticker} verfügbar.")
