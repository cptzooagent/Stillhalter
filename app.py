import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
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
        
        # Indikatoren
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        
        # Bollinger & ATR
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
            
        # Earnings-Check
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earn_str = cal.iloc[0, 0].strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")

otm_puffer_slider = st.sidebar.slider("Gewünschter Puffer (%)", 3, 25, 10)
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", 0, 100, 15)
min_stock_price, max_stock_price = st.sidebar.slider("Aktienpreis-Spanne ($)", 0, 1000, (20, 500))
only_uptrend = st.sidebar.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)

# --- KOMBI-SCAN START ---
if st.button("🚀 Kombi-Scan starten"):
    puffer_limit = otm_puffer_slider / 100 
    with st.spinner("Lade Marktliste..."):
        ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    # VIX Check
    try:
        vix_val = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        vix_status = f"VIX: {vix_val:.2f}"
    except:
        vix_status = "VIX n.a."

    for i, symbol in enumerate(ticker_liste):
        if i % 5 == 0 or i == len(ticker_liste)-1:
            progress_bar.progress((i + 1) / len(ticker_liste))
            status_text.text(f"Scanne {i+1}/{len(ticker_liste)}: {symbol}...")
        
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None or not res[1]: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if not (min_stock_price <= price <= max_stock_price): continue
            if only_uptrend and not uptrend: continue
            
            # Datums-Check (11-20 Tage)
            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 20]
            if not valid_dates:
                fallback = next((d for d in dates if (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days >= 11), None)
                valid_dates = [fallback] if fallback else []

            tk = yf.Ticker(symbol)
            best_opt_found = None
            max_y = -1

            for d_str in valid_dates:
                chain = tk.option_chain(d_str).puts
                max_strike = price * (1 - puffer_limit)
                secure_options = chain[chain['strike'] <= max_strike].sort_values('strike', ascending=False)
                
                if not secure_options.empty:
                    opt = secure_options.iloc[0]
                    bid = opt['bid'] if opt['bid'] > 0 else (opt['lastPrice'] if opt['lastPrice'] > 0 else 0.05)
                    tage = (datetime.strptime(d_str, '%Y-%m-%d') - datetime.now()).days
                    y_pa = (bid / opt['strike']) * (365 / max(1, tage)) * 100
                    
                    if y_pa > max_y:
                        max_y = y_pa
                        best_opt_found = {
                            'date': d_str, 'strike': opt['strike'], 'bid': bid, 
                            'tage': tage, 'y_pa': y_pa, 'puffer': ((price - opt['strike']) / price) * 100
                        }

            if best_opt_found and max_y >= min_yield_pa:
                all_results.append({
                    'symbol': symbol, 'price': price, 'rsi': rsi, 'uptrend': uptrend, 
                    'earn': earn, **best_opt_found
                })
        except: continue

    all_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
    status_text.empty()
    progress_bar.empty()
    
    st.markdown(f"### Marktumfeld | {vix_status}")
    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                earn_w = f" ⚠️ ER: {res['earn']}" if res['earn'] else ""
                rsi_c = "#e74c3c" if res['rsi'] > 70 else "#2ecc71" if res['rsi'] < 40 else "#555"
                with st.container(border=True):
                    st.markdown(f"**{res['symbol']}** {'✅' if res['uptrend'] else '📉'}{earn_w}")
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    st.markdown(f"""
                    <div style="font-size: 0.85em; background-color: #f1f3f6; padding: 10px; border-radius: 8px;">
                    <b>Prämie: {res['bid']:.2f}$</b> ({res['bid']*100:.0f}$)<br>
                    <b>Strike:</b> {res['strike']:.1f}$ ({res['puffer']:.1f}% Puffer)<br>
                    <b>RSI:</b> <span style="color:{rsi_c}; font-weight:bold;">{res['rsi']:.0f}</span><br>
                    <b>Termin:</b> {res['date']} ({res['tage']} Tage)
                    </div>
                    """, unsafe_allow_html=True)
