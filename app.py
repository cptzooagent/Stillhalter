import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP & STYLING ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

st.markdown("""
    <style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #1E3A8A;
        text-align: center; margin-bottom: 1.5rem;
    }
    /* Delta-Hervorhebung im Text */
    .delta-highlight { font-weight: bold; color: #1E3A8A; background-color: #e2e8f0; padding: 2px 5px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- MATHE-FUNKTIONEN ---
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

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_date, earn_str = None, ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0]
                earn_str = earn_date.strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, earn_date
    except:
        return None, [], "", 50, None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    target_prob = st.slider("🛡️ Sicherheit (OTM %)", 70, 98, 85)
    max_delta = (100 - target_prob) / 100
    min_yield_pa = st.number_input("💵 Min. Rendite p.a. (%)", value=15)
    sort_by_rsi = st.checkbox("🔄 Nach RSI sortieren", value=False)
    st.markdown("---")
    st.info("Scanner aktiv: Blue Chips & Tech")

st.markdown('<p class="main-title">🛡️ CapTrader AI Market Intelligence</p>', unsafe_allow_html=True)

# --- SCANNER ---
if st.button("🚀 Markt-Analyse starten", use_container_width=True):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "PLTR", "HOOD", "AFRM", "MSTR"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 'bid': best['bid'], 
                                        'puffer': (abs(best['strike'] - price) / price) * 100, 'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi})
            except: continue

    if results:
        df_res = pd.DataFrame(results)
        opp_df = df_res.sort_values('rsi' if sort_by_rsi else 'yield', ascending=False).head(12)
        cols = st.columns(3)
        for idx, row in enumerate(opp_df.to_dict('records')):
            with cols[idx % 3]:
                # Delta direkt in der Metrik-Anzeige
                st.metric(label=f"💰 {row['ticker']}", value=f"{row['yield']:.1f}% p.a.", delta=f"Delta {row['delta']:.2f}")
                with st.expander("Details"):
                    st.write(f"🎯 **Strike:** {row['strike']:.1f}$")
                    st.write(f"💵 **Prämie:** {row['bid']:.2f}$")
                    st.write(f"📉 **Puffer:** {row['puffer']:.1f}%")
    else:
        st.warning("Keine Treffer.")

st.markdown("---")

# --- EINZEL-CHECK (DELTA WIEDER DA) ---
st.subheader("🔍 Einzelanalyse")
c1, c2 = st.columns([1, 2])
with c1: check_mode = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: check_ticker = st.text_input("Symbol", value="NVDA").upper()

if check_ticker:
    price, dates, earn, rsi, _ = get_stock_data_full(check_ticker)
    if price and dates:
        st.markdown(f"**Kurs:** {price:.2f}$ | **RSI:** {rsi:.0f}")
        d_sel = st.selectbox("Laufzeit", dates)
        tk = yf.Ticker(check_ticker)
        try:
            chain = tk.option_chain(d_sel).puts if check_mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=check_mode), axis=1)
            
            f_df = chain[(chain['delta_calc'].abs() <= 0.4)].sort_values('strike', ascending=(check_mode == "call"))
            
            for _, opt in f_df.head(6).iterrows():
                d_abs = abs(opt['delta_calc'])
                risk_icon = "🟢" if d_abs < 0.16 else "🟡" if d_abs < 0.30 else "🔴"
                
                # Delta prominent im Titel
                with st.expander(f"{risk_icon} Strike {opt['strike']:.1f}$ | Δ {d_abs:.2f} | Bid: {opt['bid']:.2f}$"):
                    a, b, c = st.columns(3)
                    with a:
                        st.markdown(f"💰 **Einnahme:**\n{opt['bid']*100:.0f}$")
                    with b:
                        st.markdown(f"🎯 **Puffer:**\n{(abs(opt['strike']-price)/price)*100:.1f}%")
                    with c:
                        st.markdown(f"🛡️ **Delta:**\n{d_abs:.2f}")
        except: st.error("Fehler beim Abrufen der Kette.")
