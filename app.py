import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import requests
import fear_and_greed

# --- NEU: SESSION FÜR STABILITÄT ---
@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- 1. MATHE & TECHNIK (UNVERÄNDERT) ---
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

# --- 2. MARKTBREITE & SENTIMENT ---
@st.cache_data(ttl=3600)
def get_crypto_fg():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        return int(r.json()['data'][0]['value'])
    except: return 50

@st.cache_data(ttl=3600)
def get_sector_performance():
    sectors = {"XLK": "Tech", "XLY": "Consum.", "XLF": "Finanz", "XLV": "Health", "XLE": "Energy"}
    try:
        data = yf.download(list(sectors.keys()), period="1d", interval="1h", progress=False, session=get_session())['Close']
        perf = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
        return f"{sectors[perf.idxmax()]} (+{perf.max():.1f}%)", f"{sectors[perf.idxmin()]} ({perf.min():.1f}%)"
    except: return "N/A", "N/A"

# --- 3. PROFI-SCANNER LOGIK ---
def check_single_stock(symbol, p_puffer, p_min_yield, p_min_cap, min_price, max_price, only_uptrend, session):
    """Analysiert eine Aktie stabil ohne Thread-Überlastung."""
    try:
        tk = yf.Ticker(symbol, session=session)
        info = tk.info
        price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        m_cap = info.get('marketCap', 0)
        
        # Filter
        if m_cap < p_min_cap or not (min_price <= price <= max_price): return None
        
        hist = tk.history(period="150d")
        if hist.empty: return None
        
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        if only_uptrend and price < sma_200: return None
        
        # Optionen
        opt_dates = tk.options
        heute = datetime.now()
        valid_dates = [d for d in opt_dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
        if not valid_dates: return None
        
        target_date = valid_dates[0]
        chain = tk.option_chain(target_date).puts
        target_strike = price * (1 - p_puffer)
        opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] > 1)].sort_values('strike', ascending=False)
        
        if opts.empty: return None
        o = opts.iloc[0]
        
        # Preis- & Renditecheck
        bid, ask = o['bid'], o['ask']
        if bid <= 0.01 or ask > (bid * 2.5): return None
        mid = (bid + ask) / 2
        
        days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
        y_pa = (mid / o['strike']) * (365 / max(1, days_to_exp)) * 100
        
        if y_pa < p_min_yield or y_pa > 150: return None
        
        # Greeks & EM
        iv = o.get('impliedVolatility', 0.4)
        exp_move_pct = (price * (iv * np.sqrt(days_to_exp / 365)) / price) * 100
        em_safety = ((price - o['strike']) / price * 100) / exp_move_pct
        
        return {
            'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
            'puffer': (price - o['strike'])/price*100, 'bid': mid, 'rsi': rsi, 
            'tage': days_to_exp, 'status': "🛡️ Trend" if price > sma_200 else "💎 Dip",
            'delta': calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv),
            'mkt_cap': m_cap / 1e9, 'em_pct': exp_move_pct, 'em_safety': em_safety,
            'earn': tk.calendar.get('Earnings Date', [None])[0].strftime('%d.%m.') if tk.calendar is not None else "---"
        }
    except: return None


# --- SEKTION 2: DEPOT-MANAGER (STABILISIERT) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

# Assets für die Analyse
my_assets = {
    "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], 
    "ETSY": [100, 67.00], "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
    "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
    "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
}

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell im Standby.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
        depot_list = []
        progress_depot = st.progress(0)
        status_depot = st.empty()
        
        for idx, (symbol, data) in enumerate(my_assets.items()):
            status_depot.text(f"Analysiere Depot-Wert: {symbol}...")
            try:
                # Nutzt die stabilen Funktionen aus Block 1
                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: continue
                
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                qty, entry = data[0], data[1]
                perf_pct = ((price - entry) / entry) * 100

                # Strategie-Logik
                s2_d = pivots.get('S2') if pivots else None
                s2_w = pivots.get('W_S2') if pivots else None
                r2_d = pivots.get('R2') if pivots else None
                
                put_action = "⏳ Warten"
                if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"
                
                call_action = "⏳ Warten"
                if rsi > 55 and r2_d and price >= r2_d * 0.98: call_action = "🟢 JETZT (R2/RSI)"

                depot_list.append({
                    "Ticker": symbol,
                    "Earnings": earn if earn else "---",
                    "Einstand": f"{entry:.2f} $",
                    "Aktuell": f"{price:.2f} $",
                    "P/L %": f"{perf_pct:+.1f}%",
                    "RSI": int(rsi),
                    "Short Put (Repair)": put_action,
                    "Covered Call": call_action,
                    "S2 Daily": f"{s2_d:.2f} $" if s2_d else "---",
                    "S2 Weekly": f"{s2_w:.2f} $" if s2_w else "---"
                })
                time.sleep(0.8) # Sicherheits-Pause
            except: continue
            progress_depot.progress((idx + 1) / len(my_assets))
        
        st.session_state.depot_data_cache = depot_list
        status_depot.empty()
        progress_depot.empty()
        st.rerun()

else:
    # Anzeige der Depot-Tabelle
    df_depot = pd.DataFrame(st.session_state.depot_data_cache)
    st.dataframe(df_depot.style.applymap(
        lambda x: 'color: #27ae60; font-weight: bold' if 'JETZT' in str(x) or 'EXTREM' in str(x) else '',
        subset=['Short Put (Repair)', 'Covered Call']
    ), use_container_width=True)
    if st.button("🔄 Depot-Daten refreshen"):
        st.session_state.depot_data_cache = None
        st.rerun()

# --- SEKTION 3: TRADING-COCKPIT ---
st.markdown("### 🔍 Profi-Analyse & Einzelwert-Check")
symbol_input = st.text_input("Ticker Symbol", value="MU").upper()

if symbol_input:
    with st.spinner(f"Lade Dashboard für {symbol_input}..."):
        res = get_stock_data_full(symbol_input)
        if res and res[0] is not None:
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
            
            # --- SIGNAL BOX ---
            s2_d = pivots_res.get('S2') if pivots_res else None
            s2_w = pivots_res.get('W_S2') if pivots_res else None
            
            sig_text, sig_col = "⏳ Warten", "gray"
            if s2_w and price <= s2_w * 1.01: sig_text, sig_col = "🔥 EXTREM (Weekly S2)", "#ff4b4b"
            elif rsi < 35 or (s2_d and price <= s2_d * 1.02): sig_text, sig_col = "🟢 JETZT (S2/RSI)", "#27ae60"

            st.markdown(f"""
                <div style="padding:15px; border-radius:10px; border: 2px solid {sig_col}; text-align:center; background: {sig_col}10;">
                    <small style="color: {sig_col};">Strategie-Signal:</small><br>
                    <strong style="font-size:24px; color:{sig_col};">{sig_text}</strong>
                </div>
            """, unsafe_allow_html=True)

            # --- AMPEL-LOGIK ---
            ampel_col, ampel_txt = "#f1c40f", "NEUTRAL"
            if rsi < 25: ampel_col, ampel_txt = "#e74c3c", "STOPP: PANIK"
            elif rsi > 75: ampel_col, ampel_txt = "#e74c3c", "STOPP: ÜBERHITZT"
            elif rsi < 40 and uptrend: ampel_col, ampel_txt = "#27ae60", "GO: QUALITÄTS-DIP"

            st.markdown(f"""
                <div style="background-color: {ampel_col}; color: white; padding: 10px; border-radius: 8px; text-align: center; margin: 15px 0;">
                    <h3 style="margin:0;">● {ampel_txt}</h3>
                </div>
            """, unsafe_allow_html=True)

            # METRIKEN
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Preis", f"{price:.2f} $")
            m2.metric("RSI", int(rsi), delta="Tief" if rsi < 30 else None)
            m3.metric("Trend", "🟢 Bull" if uptrend else "🔴 Bear")
            m4.metric("ATR (Volatilität)", f"{atr:.2f}")

            # OPTION CHAIN
            st.markdown("#### 🎯 Option-Chain Selektor")
            mode = st.radio("Modus", ["Put (CSP)", "Call (CC)"], horizontal=True)
            valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 45]
            
            if valid_dates:
                target_d = st.selectbox("Verfallstag", valid_dates)
                chain = tk.option_chain(target_d).puts if "Put" in mode else tk.option_chain(target_d).calls
                
                # Filter & Berechnung
                days = (datetime.strptime(target_d, '%Y-%m-%d') - datetime.now()).days
                df_opt = chain[chain['openInterest'] > 10].copy()
                df_opt['Yield p.a.'] = (df_opt['bid'] / df_opt['strike']) * (365/max(1, days)) * 100
                df_opt['Puffer %'] = abs(price - df_opt['strike']) / price * 100
                
                st.dataframe(df_opt[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a.']].sort_values('strike', ascending=("Call" in mode)).head(10), use_container_width=True)

# FOOTER
st.caption(f"Letzter Scan: {datetime.now().strftime('%H:%M:%S')} | Entwickelt für CapTrader/IBKR Cash-Management")
