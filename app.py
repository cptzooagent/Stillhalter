import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time

# --- 1. INITIALISIERUNG ---
def get_tk(symbol):
    """Gibt ein Ticker-Objekt zurück."""
    return yf.Ticker(symbol)

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 2. MATHE & TECHNIK ---
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

def calculate_pivots(symbol, hist_d=None):
    """Berechnet Daily und Weekly Pivot-Punkte (optimiert)."""
    try:
        tk = get_tk(symbol)
        if hist_d is None:
            hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s2_d = p_d - (h_d - l_d)
        r2_d = p_d + (h_d - l_d) 

        # Weekly (nur wenn nötig separat)
        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: 
            return {"P": p_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_d, "W_R2": r2_d}
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        return {"P": p_d, "S2": s2_d, "R2": r2_d, "W_S2": p_w - (h_w - l_w), "W_R2": p_w + (h_w - l_w)}
    except: return None

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN"]

def get_stock_data_full(symbol):
    """Haupt-Datenbeschaffer mit fast_info Integration."""
    try:
        tk = get_tk(symbol)
        fi = tk.fast_info
        
        # fast_info Felder (Sekundenschnell)
        price = fi['last_price']
        m_cap = fi['market_cap']
        
        hist = tk.history(period="150d") 
        if hist.empty: return None
        
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        
        # Pivots berechnen
        pivots = calculate_pivots(symbol, hist_d=hist)
        
        # Earnings via calendar (langsamerer Call, aber nötig)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        return price, list(tk.options), earn_str, rsi_val, is_uptrend, m_cap, pivots
    except: return None

def get_analyst_conviction(info):
    """Analysten-Daten (benötigt das langsame info-Objekt)."""
    try:
        if not info or 'currentPrice' not in info: return "🔍 Check nötig", "#7f8c8d"
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}%)", "#9b59b6"
        elif upside > 15: return f"✅ Stark (+{upside:.0f}%)", "#27ae60"
        return f"⚖️ Neutral ({upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Scanner-Filter")
    otm_puffer_slider = st.slider("Puffer (%)", 3, 25, 15)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis ($)", 0, 1000, (60, 500))
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 20)
    only_uptrend = st.checkbox("Nur Aufwärtstrend", value=False)
    test_modus = st.checkbox("🛠️ Simulation", value=True)

# --- MARKT-MONITORING ---
st.markdown("## 🌍 Globales Markt-Monitoring")
try:
    vix = get_tk("^VIX").fast_info['last_price']
    ndx_fi = get_tk("^NDX").fast_info
    ndx_price = ndx_fi['last_price']
    dist_ndx = ((ndx_price - ndx_fi['year_high']) / ndx_fi['year_high']) * 100
except: 
    vix, ndx_price, dist_ndx = 20, 0, 0

m_color = "#e74c3c" if vix > 25 else "#27ae60"
st.markdown(f'<div style="background:{m_color};color:white;padding:15px;border-radius:10px;text-align:center;"><h3>{"ALARM" if vix > 25 else "STABIL"} (VIX: {vix:.2f})</h3></div>', unsafe_allow_html=True)
st.columns(3)[0].metric("Nasdaq 100", f"{ndx_price:,.0f}", f"{dist_ndx:.1f}%")

# --- SEKTION 2: PROFI-SCANNER (OPTIMIERT MIT FAST_INFO) ---

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Scanner analysiert Markt mit High-Speed Engine..."):
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
        progress_bar = st.progress(0)
        all_results = []

        def check_single_stock(symbol):
            try:
                tk = get_tk(symbol)
                # 1. BLITZ-CHECK (fast_info)
                fi = tk.fast_info
                price = fi['last_price']
                m_cap = fi['market_cap']
                
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): 
                    return None
                
                # 2. TECHNIK-CHECK (history)
                res_full = get_stock_data_full(symbol)
                if not res_full: return None
                _, dates, earn, rsi, uptrend, _, pivots = res_full
                
                if only_uptrend and not uptrend: return None
                
                # 3. OPTIONEN-FILTERUNG
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] > 1)].sort_values('strike', ascending=False)
                
                if opts.empty: return None
                o = opts.iloc[0]

                # RENDITE & RISIKO
                fair_price = (o['bid'] + o['ask']) / 2 if o['bid'] > 0 else o['lastPrice']
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                
                if y_pa > 150 or y_pa < p_min_yield: return None

                iv = o.get('impliedVolatility', 0.4)
                exp_move_pct = (iv * np.sqrt(days_to_exp / 365)) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                
                # 4. LAZY LOADING FÜR ANALYSTEN (Nur wenn Trade-Kandidat!)
                info = tk.info # Hier einmalig der langsame Call
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                stars_count = 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1

                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'bid': fair_price, 'rsi': rsi, 'earn': earn if earn else "---", 
                    'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 
                    'delta': abs(delta_val), 'sent_icon': "🟢", 
                    'stars_str': "⭐" * stars_count,
                    'analyst_label': analyst_txt, 'analyst_color': analyst_col, 
                    'mkt_cap': m_cap / 1e9, 'em_pct': exp_move_pct, 'em_safety': em_safety
                }
            except: return None

        # PARALLELE VERARBEITUNG
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(ticker_liste))

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
        st.rerun()

# --- ANZEIGE DER KACHELN ---
if st.session_state.profi_scan_results:
    res_list = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups ({len(res_list)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    
    for idx, res in enumerate(res_list):
        with cols[idx % 4]:
            is_earning_risk = False
            earn_str = res.get('earn', "---")
            if earn_str != "---":
                try:
                    parts = earn_str.split('.')
                    earn_date = datetime(heute_dt.year, int(parts[1]), int(parts[0]))
                    if earn_date < heute_dt: earn_date = datetime(heute_dt.year + 1, int(parts[1]), int(parts[0]))
                    if 0 <= (earn_date - heute_dt).days <= 14: is_earning_risk = True
                except: pass

            card_border = "2px solid #ef4444" if is_earning_risk else "1px solid #e5e7eb"
            html_code = f"""
            <div style="background: white; border: {card_border}; border-radius: 12px; padding: 15px; margin-bottom: 15px; height: 500px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 800; font-size: 1.1em;">{res['symbol']} {res['stars_str']}</span>
                    <span style="font-size: 0.8em; color: #6b7280;">{res['status']}</span>
                </div>
                <div style="margin: 15px 0; text-align: center;">
                    <div style="font-size: 0.7em; color: #6b7280; text-transform: uppercase;">Yield p.a.</div>
                    <div style="font-size: 2em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85em;">
                    <div>Strike: <b>{res['strike']:.1f}$</b></div>
                    <div>Puffer: <b>{res['puffer']:.1f}%</b></div>
                    <div>Delta: <b style="color: #10b981;">{res['delta']:.2f}</b></div>
                    <div>Tage: <b>{res['tage']}d</b></div>
                </div>
                <div style="background: #f9fafb; padding: 10px; border-radius: 8px; margin-top: 15px; font-size: 0.8em;">
                    EM Sicherheit: <b>{res['em_safety']:.1f}x</b><br>
                    RSI: <b>{int(res['rsi'])}</b> | Cap: <b>{res['mkt_cap']:.0f}B</b><br>
                    Earnings: <b style="color: {'#ef4444' if is_earning_risk else '#111827'};">{res['earn']}</b>
                </div>
                <div style="margin-top: 15px; padding: 8px; border-radius: 6px; background: {res['analyst_color']}20; color: {res['analyst_color']}; font-weight: bold; font-size: 0.75em; text-align: center;">
                    {res['analyst_label']}
                </div>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)

# --- SEKTION 2: DEPOT-MANAGER (OPTIMIERT) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Depot-Analyse bereit.")
    if st.button("🚀 Depot jetzt analysieren", use_container_width=True):
        with st.spinner("Lade Bestandsdaten via Fast-Engine..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], 
                "ETSY": [100, 67.00], "GTLB": [100, 41.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "NVO": [100, 97.00], "PLTR": [100, 35.00],
                "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    tk = get_tk(symbol)
                    fi = tk.fast_info
                    price = fi['last_price']
                    
                    # Technik-Daten (History)
                    hist = tk.history(period="60d")
                    rsi_val = calculate_rsi(hist['Close']).iloc[-1]
                    pivots = calculate_pivots(symbol, hist_d=hist)
                    
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100
                    
                    # Reparatur-Signale
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    
                    put_action = "⏳ Warten"
                    if rsi_val < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"

                    depot_list.append({
                        "Ticker": symbol,
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "RSI": int(rsi_val),
                        "Repair Put": put_action,
                        "Weekly S2": f"{s2_w:.2f} $" if s2_w else "n.a."
                    })
                except: continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    col_header, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Refresh"):
            st.session_state.depot_data_cache = None
            st.rerun()
    st.table(pd.DataFrame(st.session_state.depot_data_cache))

# --- SEKTION 3: PROFI-COCKPIT ---
st.markdown("### 🔍 Einzelwert-Analyse")
symbol_input = st.text_input("Ticker Symbol", value="NVDA").upper()

if symbol_input:
    try:
        tk = get_tk(symbol_input)
        fi = tk.fast_info
        price = fi['last_price']
        
        res_full = get_stock_data_full(symbol_input)
        if res_full:
            price, dates, earn, rsi, uptrend, m_cap, pivots_res = res_full
            
            # Ampel-Logik
            ampel_color = "#27ae60" if (uptrend and 35 < rsi < 65) else "#f1c40f"
            if rsi < 30 or rsi > 70: ampel_color = "#e74c3c"
            
            st.markdown(f"""
                <div style="background:{ampel_color}; color:white; padding:15px; border-radius:10px; text-align:center;">
                    <h2 style="margin:0;">RSI: {int(rsi)} | {'STABIL' if ampel_color == "#27ae60" else 'VORSICHT'}</h2>
                </div>
            """, unsafe_allow_html=True)

            # Option Chain
            st.markdown("#### 🎯 Option-Selection")
            valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 60]
            if valid_dates:
                t_date = st.selectbox("Verfallstag", valid_dates)
                chain = tk.option_chain(t_date).puts
                chain['mid'] = (chain['bid'] + chain['ask']) / 2
                chain['Puffer %'] = ((price - chain['strike']) / price) * 100
                
                # Rendite p.a.
                days = (datetime.strptime(t_date, '%Y-%m-%d') - datetime.now()).days
                chain['Yield p.a. %'] = (chain['mid'] / chain['strike']) * (365 / max(1, days)) * 100
                
                output = chain[chain['strike'] < price][['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].sort_values('strike', ascending=False).head(10)
                st.dataframe(output.style.format({'Puffer %': '{:.1f}%', 'Yield p.a. %': '{:.1f}%'}), use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")

st.divider()
st.caption(f"Engine: Fast-Info Hybrid | Letztes Update: {datetime.now().strftime('%H:%M:%S')}")
