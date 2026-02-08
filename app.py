import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHEMATIK (BSM) ---
def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    try:
        if T <= 0 or sigma <= 0 or S <= 0: return 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1)
    except:
        return 0

def calculate_rsi(data, window=14):
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FETCH (ROBUST) ---
@st.cache_data(ttl=900)
def get_stock_info(symbol):
    """Holt alle Basisdaten für einen Ticker sicher."""
    try:
        tk = yf.Ticker(symbol)
        # Fallback für Preis
        fast = tk.fast_info
        price = fast.get('last_price') or tk.history(period="1d")['Close'].iloc[-1]
        
        hist = tk.history(period="1mo")
        rsi_val = calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        
        # Earnings
        earn_str = ""
        earn_date = None
        if tk.calendar is not None and 'Earnings Date' in tk.calendar:
            earn_date = tk.calendar['Earnings Date'][0]
            earn_str = earn_date.strftime('%d.%m.')
            
        return {
            "price": price,
            "rsi": rsi_val,
            "earn_str": earn_str,
            "earn_date": earn_date,
            "options": list(tk.options)
        }
    except Exception as e:
        return None

# --- UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")

# SEKTION 1: SCANNER
if st.button("🚀 Kombi-Scan starten"):
    watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "COIN", "MSTR", "AMD", "META", "AMZN", "GOOGL"] # Gekürzte Liste zum Testen
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        data = get_stock_info(t)
        
        if data and data['options']:
            try:
                tk = yf.Ticker(t)
                # Nächstgelegenes Datum zu 30 Tagen finden
                target_date = min(data['options'], key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                # Delta Berechnung
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(data['price'], r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                
                # Filter
                matches = chain[chain['delta_val'].abs() <= max_delta].copy()
                if not matches.empty:
                    days = max(1, T * 365)
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (365 / days) * 100
                    best = matches[matches['y_pa'] >= min_yield_pa].sort_values('y_pa', ascending=False)
                    
                    if not best.empty:
                        row = best.iloc[0]
                        results.append({
                            'ticker': t, 'yield': row['y_pa'], 'strike': row['strike'], 
                            'bid': row['bid'], 'puffer': (abs(row['strike'] - data['price']) / data['price']) * 100, 
                            'delta': abs(row['delta_val']), 'rsi': data['rsi'], 'earn': data['earn_str']
                        })
            except: continue

    if results:
        st.session_state['scan_results'] = results
    else:
        st.warning("Keine Treffer mit aktuellen Filtern.")

# Anzeige Scan-Ergebnisse (aus Session State)
if 'scan_results' in st.session_state:
    df_res = pd.DataFrame(st.session_state['scan_results'])
    cols = st.columns(4)
    for idx, row in enumerate(df_res.to_dict('records')):
        with cols[idx % 4]:
            st.metric(f"{row['ticker']}", f"{row['yield']:.1f}% p.a.")
            st.write(f"Strike: **{row['strike']}$** | Δ: {row['delta']:.2f}")
            st.caption(f"Puffer: {row['puffer']:.1f}% | RSI: {row['rsi']:.0f}")

st.divider()

# SEKTION 2: DEPOT (STABILISIERT)
st.subheader("💼 Smart Depot-Manager")
depot_list = ["AFRM", "HOOD", "NVDA"] # Beispielhaft
d_cols = st.columns(3)

for i, ticker in enumerate(depot_list):
    info = get_stock_info(ticker)
    if info:
        with d_cols[i % 3]:
            with st.expander(f"{ticker} - {info['price']:.2f}$", expanded=True):
                st.write(f"RSI: {info['rsi']:.0f}")
                if info['earn_str']:
                    st.warning(f"Earnings: {info['earn_str']}")
    else:
        st.error(f"Datenfehler: {ticker}")
