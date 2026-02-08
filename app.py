import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE-FUNKTIONEN ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except:
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0
            
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
            
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and not cal.empty:
                # Handhabung verschiedener yfinance-Formate
                if hasattr(cal, 'iloc'):
                    earn_str = cal.iloc[0, 0].strftime('%d.%m.')
                else:
                    earn_str = cal[0].strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

# --- 3. UI SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
otm_puffer_slider = st.sidebar.slider("Gewünschter Puffer (%)", 3, 25, 10)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", 0, 100, 15)
min_stock_p, max_stock_p = st.sidebar.slider("Aktienpreis-Spanne ($)", 0, 1000, (20, 500))
only_uptrend = st.sidebar.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    puffer_limit = otm_puffer_slider / 100 
    with st.spinner("Lade Marktliste..."):
        ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    # VIX Wetter
    try:
        v_val = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        v_color = "#2ecc71" if v_val < 20 else "#f1c40f" if v_val < 30 else "#e74c3c"
        v_status = f"<span style='color:{v_color}; font-weight:bold;'>VIX: {v_val:.2f}</span>"
    except:
        v_status = "VIX n.a."

    for i, symbol in enumerate(ticker_liste):
        if i % 5 == 0 or i == len(ticker_liste)-1:
            progress_bar.progress((i + 1) / len(ticker_liste))
            status_text.text(f"Scanne {i+1}/{len(ticker_liste)}: {symbol}...")
        
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None or not res[1]: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if not (min_stock_p <= price <= max_stock_p): continue
            if only_uptrend and not uptrend: continue
            
            # 11-20 Tage Fenster
            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 20]
            if not valid_dates:
                fb = next((d for d in dates if (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days >= 11), None)
                valid_dates = [fb] if fb else []

            tk = yf.Ticker(symbol)
            best_opt = None
            max_y = -1

            for d_str in valid_dates:
                chain = tk.option_chain(d_str).puts
                strike_limit = price * (1 - puffer_limit)
                options = chain[chain['strike'] <= strike_limit].sort_values('strike', ascending=False)
                
                if not options.empty:
                    o = options.iloc[0]
                    bid = o['bid'] if o['bid'] > 0 else (o['lastPrice'] if o['lastPrice'] > 0 else 0.05)
                    tage = (datetime.strptime(d_str, '%Y-%m-%d') - datetime.now()).days
                    y_pa = (bid / o['strike']) * (365 / max(1, tage)) * 100
                    
                    if y_pa > max_y:
                        max_y = y_pa
                        best_opt = {
                            'date': d_str, 'strike': o['strike'], 'bid': bid, 
                            'tage': tage, 'y_pa': y_pa, 'puffer': ((price - o['strike']) / price) * 100
                        }

            if best_opt and max_y >= min_yield_pa:
                all_results.append({
                    'symbol': symbol, 'price': price, 'rsi': rsi, 'uptrend': uptrend, 
                    'earn': earn, **best_opt
                })
        except: continue

    all_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
    status_text.empty()
    progress_bar.empty()
    
    st.markdown(f"### Marktumfeld | {v_status}", unsafe_allow_html=True)
    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        st.success(f"{len(all_results)} Chancen gefunden.")
        cols = st.columns(4)
        for idx, r in enumerate(all_results):
            with cols[idx % 4]:
                earn_icon = f" ⚠️ ER: {r['earn']}" if r['earn'] else ""
                rsi_c = "#e74c3c" if r['rsi'] > 70 else "#2ecc71" if r['rsi'] < 40 else "#555"
                with st.container(border=True):
                    st.markdown(f"**{r['symbol']}** {'✅' if r['uptrend'] else '📉'}{earn_icon}")
                    st.metric("Yield p.a.", f"{r['y_pa']:.1f}%")
                    st.markdown(f"""
                    <div style="font-size: 0.85em; background-color: #f1f3f6; padding: 10px; border-radius: 8px;">
                    <b>Prämie: {r['bid']:.2f}$</b> ({r['bid']*100:.0f}$)<br>
                    <b>Strike:</b> {r['strike']:.1f}$ ({r['puffer']:.1f}% Puffer)<br>
                    <b>RSI:</b> <span style="color:{rsi_c}; font-weight:bold;">{r['rsi']:.0f}</span><br>
                    <b>Datum:</b> {r['date']} ({r['tage']} T.)
                    </div>
                    """, unsafe_allow_html=True)

# --- SEKTION 2: DEPOT ---
st.markdown("---")
st.markdown("### 💼 Smart Depot-Manager")
depot_data = [
    {'Ticker': 'AFRM', 'Einstand': 76.00}, {'Ticker': 'HOOD', 'Einstand': 120.0},
    {'Ticker': 'JKS', 'Einstand': 50.00}, {'Ticker': 'HIMS', 'Einstand': 37.00},
    {'Ticker': 'NVO', 'Einstand': 97.00}, {'Ticker': 'TTD', 'Einstand': 102.00}
]
d_cols = st.columns(4)
for i, d_item in enumerate(depot_data):
    p, _, e, rs, ut, _, _ = get_stock_data_full(d_item['Ticker'])
    if p:
        df_p = (p / d_item['Einstand'] - 1) * 100
        with d_cols[i % 4]:
            with st.container(border=True):
                c_perf = "#2ecc71" if df_p >= 0 else "#e74c3c"
                st.markdown(f"**{d_item['Ticker']}** <span style='float:right; color:{c_perf};'>{df_p:+.1f}%</span>", unsafe_allow_html=True)
                st.caption(f"Kurs: {p:.2f}$ | RSI: {rs:.0f}")
                if rs > 65: st.success("🟢 Call-Chance!")
                if e: st.warning(f"📅 ER: {e}")

# --- SEKTION 3: EINZEL-CHECK ---
st.markdown("---")
st.subheader("🔍 Einzel-Check")
e_c1, e_c2 = st.columns([1, 2])
with e_c1: e_mode = st.radio("Typ", ["put", "call"], horizontal=True)
with e_c2: e_ticker = st.text_input("Symbol", value="NVDA").upper()
if e_ticker:
    p, dts, e, rs, ut, _, _ = get_stock_data_full(e_ticker)
    if p:
        st.write(f"Kurs: {p:.2f}$ | RSI: {rs:.0f} | Trend: {'✅' if ut else '📉'}")
        if dts:
            s_dt = st.selectbox("Laufzeit", dts)
            tk_obj = yf.Ticker(e_ticker)
            ch = tk_obj.option_chain(s_dt).puts if e_mode == "put" else tk_obj.option_chain(s_dt).calls
            st.dataframe(ch[['strike', 'bid', 'ask', 'volume', 'impliedVolatility']].head(10))
