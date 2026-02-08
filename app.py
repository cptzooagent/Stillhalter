import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader Pro Scanner", layout="wide")

# Styling für bessere Lesbarkeit
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stMetric { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MATHE: DELTA & RSI ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FUNKTIONEN (OPTIMIERT) ---
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        info = tk.fast_info
        price = info['last_price']
        
        # Historie für RSI und Volumen-Check
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        # Volumen-Verhältnis (Aktuell vs. Durchschnitt)
        vol_ratio = 1.0
        if len(hist) > 1:
            avg_vol = hist['Volume'].mean()
            current_vol = hist['Volume'].iloc[-1]
            vol_ratio = current_vol / avg_vol

        earn_str = ""
        earn_dt = None
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_dt = cal['Earnings Date'][0]
                earn_str = earn_dt.strftime('%d.%m.')
        except: pass
        
        return price, list(tk.options), earn_str, rsi_val, earn_dt, vol_ratio
    except:
        return None, [], "", 50, None, 1.0

# --- UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Fokus")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 99, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=15)
st.sidebar.markdown("---")
st.sidebar.caption("💡 Tipp: Kombiniere niedrigen RSI (<35) mit starken Support-Zonen im Chart.")

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Markt-Scan starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "PLTR", "HOOD", "AFRM", "COIN", "SQ", "AMD", "NFLX"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _, vol_ratio = get_stock_data_full(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                # Finde das Datum, das am nächsten an 30 Tagen liegt
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = max(1/365, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365)
                
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, T * 365)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    best = safe_opts.sort_values('y_pa', ascending=False).iloc[0]
                    
                    if best['y_pa'] >= min_yield_pa:
                        results.append({
                            'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 
                            'bid': best['bid'], 'puffer': (abs(best['strike'] - price) / price) * 100, 
                            'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi, 'vol': vol_ratio
                        })
            except: continue

    if results:
        res_df = pd.DataFrame(results).sort_values('yield', ascending=False)
        cols = st.columns(3)
        for idx, row in enumerate(res_df.to_dict('records')):
            with cols[idx % 3]:
                # Farbindikator für RSI
                rsi_color = "🔵" if row['rsi'] < 35 else "🔴" if row['rsi'] > 65 else "⚪"
                vol_status = "🔥" if row['vol'] > 1.5 else ""
                
                with st.container():
                    st.markdown(f"### {row['ticker']} {rsi_color} {vol_status}")
                    st.metric("Rendite p.a.", f"{row['yield']:.1f}%", f"Δ {row['delta']:.2f}", delta_color="inverse")
                    st.write(f"**Strike:** {row['strike']:.1f}$ | **Puffer:** {row['puffer']:.1f}%")
                    if row['earn']: st.warning(f"📅 Earnings: {row['earn']}")
                    st.caption(f"RSI: {row['rsi']:.0f} | Vol-Index: {row['vol']:.1f}")
    else:
        st.info("Keine Optionen gefunden, die den Kriterien entsprechen.")

# --- SEKTION 2: DEPOT ---
st.write("---")
st.subheader("💼 Depot-Status & Signale")
depot_list = ["AFRM", "HOOD", "NVDA", "PLTR", "META"]
d_cols = st.columns(len(depot_list))

for i, t in enumerate(depot_list):
    price, _, earn, rsi, earn_dt, _ = get_stock_data_full(t)
    if price:
        with d_cols[i]:
            st.markdown(f"**{t}**")
            st.write(f"{price:.1f}$")
            if rsi < 35: st.info(f"RSI {rsi:.0f}\nBuy?")
            elif rsi > 65: st.success(f"RSI {rsi:.0f}\nSell?")
            else: st.text(f"RSI {rsi:.0f}")

# --- SEKTION 3: EINZEL-CHECK ---
st.write("---")
st.subheader("🔍 Deep-Dive Einzel-Check")
ticker_in = st.text_input("Symbol prüfen", "NVDA").upper()

if ticker_in:
    price, dates, earn, rsi, _, vol = get_stock_data_full(ticker_in)
    if price:
        st.write(f"Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}** | Volumen-Faktor: **{vol:.1f}x**")
        expiry = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(ticker_in)
        chain = tk.option_chain(expiry).puts
        T = max(1/365, (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365)
        
        chain['delta_calc'] = chain.apply(lambda o: calculate_bsm_delta(price, o['strike'], T, o['impliedVolatility'] or 0.4), axis=1)
        
        # Zeige nur relevante Strikes nahe am Markt
        for _, opt in chain[chain['delta_calc'].abs() < 0.4].sort_values('strike', ascending=False).head(6).iterrows():
            d_abs = abs(opt['delta_calc'])
            label = f"Strike {opt['strike']:.1f}$ | Delta: {d_abs:.2f} | Bid: {opt['bid']:.2f}$"
            with st.expander(label):
                st.write(f"💰 **Prämie:** {opt['bid']*100:.0f}$ Cash-Einnahme")
                st.write(f"📉 **Puffer zum Kurs:** {abs(opt['strike']-price)/price*100:.1f}%")
                st.write(f"🌊 **Implizite Vola:** {int((opt['impliedVolatility'] or 0)*100)}%")
