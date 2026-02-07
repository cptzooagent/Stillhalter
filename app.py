import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time

# --- 1. SETUP & MATHE ---
st.set_page_config(page_title="CapTrader AI Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Präzise Delta-Berechnung mit Schutz gegen Nullwerte."""
    T = max(T, 0.0001)
    # Nutzt 40% Vola als Standard, falls Yahoo 0.0 liefert
    sig = sigma if (sigma and sigma > 0.1) else 0.4
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        if option_type == 'call':
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1)
    except:
        return 0.0

@st.cache_data(ttl=900)
def get_stock_basics(symbol):
    """Holt Kurs, Optionen und Earnings-Termine."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_info = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_info = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_info
    except:
        return None, [], ""

# --- 2. SIDEBAR (STRATEGIE) ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.sidebar.subheader("💰 Preis-Filter")
min_stock_p = st.sidebar.number_input("Mindestkurs ($)", value=40)
max_stock_p = st.sidebar.number_input("Maximalkurs ($)", value=600)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. MARKT-SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "MSTR", "UBER", "DIS", "PYPL", "AFRM", "SQ", "RIVN"]
    results = []
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_basics(t)
        if price and min_stock_p <= price <= max_stock_p and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                days = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1)
                T = days / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility']), axis=1)
                matches = chain[(chain['delta_val'].abs() <= max_delta) & (chain['bid'] > 0)].copy()
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (365 / days) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    if best['y_pa'] >= min_yield_pa:
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'price': price, 'earn': earn})
            except: continue
    if results:
        st.table(pd.DataFrame(results).sort_values('yield', ascending=False))
    else: st.warning("Keine Treffer gefunden.")

st.divider()

# --- 4. DEPOT-STATUS ---
st.subheader("💼 Depot-Status")
depot_data = [
    {"T": "AFRM", "E": 76.0}, {"T": "ELF", "E": 109.0}, {"T": "HOOD", "E": 120.0}, 
    {"T": "TTD", "E": 102.0}, {"T": "SE", "E": 170.0}, {"T": "RBRK", "E": 70.0}
]
d_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn = get_stock_basics(item['T'])
    if price:
        perf = (price / item['E'] - 1) * 100
        with d_cols[i % 3]:
            st.metric(f"{item['T']}", f"{price:.2f}$", f"{perf:.1f}%", delta_color="normal" if perf > -15 else "inverse")
            if earn: st.caption(f"Earnings: {earn}")

st.divider()

# --- 5. EINZEL-CHECK (REAKTIONS-FIX) ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_input = st.text_input("Ticker Symbol", value="ELF").upper().strip()

if t_input:
    # Daten für den Ticker holen
    price, dates, earn = get_stock_basics(t_input)
    
    if price and dates:
        if earn: st.info(f"📅 Nächste Earnings für {t_input}: {earn}")
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        
        # KEY-FIX: Sorgt dafür, dass die Box sich bei Ticker-Wechsel leert/neu lädt
        d_sel = st.selectbox("Laufzeit wählen", dates, key=f"select_{t_input}")
        
        try:
            tk = yf.Ticker(t_input)
            chain = tk.option_chain(d_sel)
            df = chain.puts if mode == "put" else chain.calls
            T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
            
            # Sortierung & Filter
            if mode == "put":
                # Puts: Strikes unter dem Kurs, abwärts sortiert (80, 75, 70...)
                df = df[df['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                # Calls: Strikes über dem Kurs, aufwärts sortiert (85, 90, 95...)
                df = df[df['strike'] >= price * 0.95].sort_values('strike', ascending=True)
            
            for _, opt in df.head(8).iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'], mode)
                d_abs = abs(delta)
                
                # Risiko-Ampel
                is_itm = (mode == "put" and opt['strike'] > price) or (mode == "call" and opt['strike'] < price)
                icon = "🔴 ITM" if is_itm else "🟢 OTM" if d_abs < 0.20 else "🟡 NEAR"
                
                with st.expander(f"{icon} | Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$ | Delta: {d_abs:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📊 **OTM-Chance:** {(1-d_abs)*100:.1f}%")
                        st.write(f"💰 **Einnahme:** {opt['bid']*100:.0f}$")
                    with col2:
                        st.write(f"🎯 **Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                        st.write(f"💼 **Kapital:** {opt['strike']*100:,.0f}$")
        except:
            st.warning(f"Konnte Optionsdaten für {t_input} ({d_sel}) nicht laden.")
