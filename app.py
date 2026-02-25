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

# ==========================================
# --- BLOCK 1: PROFI-SCANNER (FINALE VERSION) ---
# ==========================================

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Scanner analysiert Markt & News..."):
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
        progress_bar = st.progress(0)
        all_results = []

        def check_single_stock(symbol):
            try:
                tk = get_tk(symbol)
                info = tk.info
                if not info or 'currentPrice' not in info: return None
                
                price = info.get('currentPrice', 0)
                m_cap = info.get('marketCap', 0)
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): return None
                
                res_full = get_stock_data_full(symbol)
                if not res_full or res_full[0] is None: return None
                
                # Entpacken (angepasst an deine get_stock_data_full)
                _, dates, earn, rsi, uptrend, _, atr, pivots = res_full
                sent_status, sent_msg, sent_score = get_openclaw_analysis(symbol)
                
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] > 0)].sort_values('strike', ascending=False)
                
                if opts.empty: return None
                o = opts.iloc[0]

                fair_price = (o['bid'] + o['ask']) / 2
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                if y_pa < p_min_yield: return None

                iv = o.get('impliedVolatility', 0.4)
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                
                # Wachstums-Label (Lila Box Daten)
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                # EM Berechnung
                em_val = 1.25 * iv * price * (days_to_exp/365)**0.5
                em_pct = (em_val / price) * 100

                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': ((price - o['strike']) / price) * 100, 'bid': fair_price, 
                    'rsi': rsi, 'earn': earn, 'tage': days_to_exp, 
                    'status': "Trend" if uptrend else "Dip", 
                    'delta': abs(delta_val), 
                    'sent_icon': "🟢" if sent_status == "Bullish" else "🔴" if sent_status == "Bearish" else "🟡",
                    'sent_status': sent_status, 'news_snippet': sent_msg,
                    'stars_str': "⭐" * (3 if "HYPER" in analyst_txt or "Stark" in analyst_txt else 2),
                    'analyst_label': analyst_txt, 'analyst_color': analyst_col,
                    'em_pct': em_pct, 'em_safety': (price - o['strike']) / em_val if em_val > 0 else 1.0
                }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(ticker_liste))

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
        st.rerun()

# --- ANZEIGE (BÜNDIG LINKS KOPIEREN) ---
if st.session_state.profi_scan_results:
    res_list = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(res_list)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    
    for idx, res in enumerate(res_list):
        with cols[idx % 4]:
            # News-Logik
            s_status = res.get('sent_status', 'Neutral')
            n_color = "#27ae60" if s_status == "Bullish" else "#e74c3c" if s_status == "Bearish" else "#f59e0b"
            
            # Earnings-Warner (Rote Umrandung Bild 2)
            is_risk = False
            e_str = res.get('earn', "---")
            if e_str and e_str != "---":
                try:
                    d, m = map(int, e_str.split('.'))
                    if 0 <= (datetime(2026, m, d) - heute_dt).days <= res.get('tage', 30): is_risk = True
                except: pass

            b_style = "3px solid #ef4444" if is_risk else "1px solid #e5e7eb"
            s_style = "0 10px 15px -3px rgba(239, 68, 68, 0.15)" if is_risk else "0 4px 6px -1px rgba(0, 0, 0, 0.1)"

            html_code = f"""
<div style="background: white; border: {b_style}; border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: {s_style}; font-family: sans-serif; height: 640px; display: flex; flex-direction: column;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.4em; font-weight: 800; color: #111827;">{res.get('symbol')} <span style="color: #f59e0b; font-size: 0.8em;">{res.get('stars_str', '⭐')}</span></span>
<span style="font-size: 0.75em; font-weight: 700; color: #3b82f6; background: #ebf5ff; padding: 4px 10px; border-radius: 8px;">{res.get('status')}</span>
</div>
<div style="margin: 12px 0;">
<div style="font-size: 0.7em; color: #6b7280; font-weight: 700; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 2.2em; font-weight: 900; color: #111827; line-height: 1;">{res.get('y_pa', 0):.1f}%</div>
</div>
<div style="background: {n_color}10; border-left: 4px solid {n_color}; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
<div style="font-size: 0.75em; font-weight: 800; color: {n_color}; margin-bottom: 4px;">{res.get('sent_icon')} OPENCLAW: {s_status.upper()}</div>
<div style="font-size: 0.7em; color: #374151; line-height: 1.4; height: 3.2em; overflow: hidden;">{res.get('news_snippet', 'Keine News')}</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 15px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 10px;"><div style="font-size: 0.65em; color: #6b7280; font-weight: 600;">Strike</div><div style="font-size: 1em; font-weight: 700;">{res.get('strike', 0):.1f}$</div></div>
<div style="border-left: 3px solid #f59e0b; padding-left: 10px;"><div style="font-size: 0.65em; color: #6b7280; font-weight: 600;">Mid</div><div style="font-size: 1em; font-weight: 700;">{res.get('bid', 0):.2f}$</div></div>
<div style="border-left: 3px solid #3b82f6; padding-left: 10px;"><div style="font-size: 0.65em; color: #6b7280; font-weight: 600;">Puffer</div><div style="font-size: 1em; font-weight: 700;">{res.get('puffer', 0):.1f}%</div></div>
<div style="border-left: 3px solid #10b981; padding-left: 10px;"><div style="font-size: 0.65em; color: #6b7280; font-weight: 600;">Delta</div><div style="font-size: 1em; font-weight: 700; color: #10b981;">{res.get('delta', 0):.2f}</div></div>
</div>
<div style="background: #fffbeb; border: 1px dashed #f59e0b; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
<div style="display: flex; justify-content: space-between; font-size: 0.7em; font-weight: 700;">
<span style="color: #92400e;">Stat. Erwartung (EM):</span><span style="color: #b45309;">±{res.get('em_pct', 0):.1f}%</span>
</div>
<div style="font-size: 0.65em; color: #b45309; margin-top: 2px;">Sicherheit: <b>{res.get('em_safety', 1):.1f}x EM</b></div>
</div>
<div style="margin-top: auto;">
<div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #4b5563; margin-bottom: 12px; background: #f9fafb; padding: 6px 10px; border-radius: 8px;">
<span>⏳ <b>{res.get('tage')}d</b></span>
<span style="font-weight: 700;">RSI: {int(res.get('rsi', 0))}</span>
<span style="font-weight: 800; color: {'#ef4444' if is_risk else '#6b7280'};">{'⚠️' if is_risk else '🗓️'} {e_str}</span>
</div>
<div style="background: {res.get('analyst_color', '#9b59b6')}15; color: {res.get('analyst_color', '#9b59b6')}; padding: 12px; border-radius: 10px; font-size: 0.75em; font-weight: 800; text-align: center; border-left: 5px solid {res.get('analyst_color', '#9b59b6')};">
{res.get('analyst_label', 'WACHSTUM ANALYSIEREN')}
</div>
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

# ==========================================
# --- BLOCK 3: PROFI-ANALYSE & COCKPIT ---
# ==========================================

st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")

symbol_input = st.text_input("Ticker Symbol", value="MU").upper()

if symbol_input:
    try:
        with st.spinner(f"Lade Daten für {symbol_input}..."):
            tk = get_tk(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)

            if res and res[0] is not None:
                # Unpack (price, dates, earn, rsi, uptrend, m_cap, pivots)
                price, dates, earn, rsi, uptrend, m_cap, pivots_res = res
                analyst_txt, analyst_col = get_analyst_conviction(info)

                # 1. EARNINGS & FUNDAMENTALS (Lila Akzent wie im Bild)
                st.markdown(f"""
                    <div style="background: #f8fafc; border-left: 5px solid #8b5cf6; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                        <span style="font-weight: 700; color: #1f2937;">💡 Analysten-Einschätzung:</span> 
                        <span style="color: #8b5cf6; font-weight: 800;">{analyst_txt}</span><br>
                        <span style="font-size: 0.85em; color: #64748b;">🗓️ Nächste Earnings: <b>{earn if earn else "---"}</b> | RSI: {int(rsi)} | Trend: {"✅" if uptrend else "❌"}</span>
                    </div>
                """, unsafe_allow_html=True)

                # 2. SIGNAL-AMPEL (Die farbige Box aus deinem Code)
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
                
                put_action_scanner = "⏳ Warten"
                signal_color = "#f1c40f" # Gelb (Standard)

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b" # Rot
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60" # Grün

                st.markdown(f"""
                    <div style="background-color: {signal_color}; color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
                        <h2 style="margin:0; font-size: 2em; font-weight: 800;">{put_action_scanner}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # 3. PIVOT-KACHELN (Zusatz für die technische Absicherung)
                if pivots_res:
                    pk1, pk2, pk3, pk4 = st.columns(4)
                    pk1.metric("Pivot (P)", f"{pivots_res['P']:.2f} $")
                    pk2.metric("Support S1", f"{(pivots_res['P']*0.98):.2f} $")
                    pk3.metric("Daily S2", f"{s2_d:.2f} $")
                    pk4.metric("Weekly Boden", f"{s2_w:.2f} $")

                # 4. OPTION CHAIN SEKTION
                st.markdown("---")
                option_mode = st.radio("Strategie:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                # Verfallstage filtern (5 bis 45 Tage)
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 45]
                
                if valid_dates:
                    target_date = st.selectbox("Verfallstag wählen", valid_dates)
                    days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                    
                    opt_chain = tk.option_chain(target_date)
                    df_disp = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    
                    # Kopie für Berechnungen
                    df_disp = df_disp[df_disp['openInterest'] > 5].copy()
                    
                    if "Put" in option_mode:
                        df_disp = df_disp[df_disp['strike'] < price]
                        sort_asc = False
                    else:
                        df_disp = df_disp[df_disp['strike'] > price]
                        sort_asc = True
                    
                    # Rendite p.a. Berechnung
                    df_disp['Mid'] = (df_disp['bid'] + df_disp['ask']) / 2
                    df_disp['Yield p.a. %'] = (df_disp['Mid'] / df_disp['strike']) * (365 / max(1, days_to_exp)) * 100
                    df_disp['Puffer %'] = abs((price - df_disp['strike']) / price) * 100

                    # Anzeige der Top 10
                    st.dataframe(
                        df_disp[['strike', 'bid', 'ask', 'Mid', 'Puffer %', 'Yield p.a. %', 'openInterest', 'impliedVolatility']]
                        .sort_values('strike', ascending=sort_asc)
                        .head(10)
                        .style.format({
                            'strike': '{:.2f}$', 'bid': '{:.2f}', 'ask': '{:.2f}', 'Mid': '{:.2f}',
                            'Puffer %': '{:.1f}%', 'Yield p.a. %': '{:.1f}%', 'impliedVolatility': '{:.1%}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("Keine passenden Verfallstage (5-45 Tage) gefunden.")

    except Exception as e:
        if "Too Many Requests" in str(e):
            st.error("🚫 Yahoo-Sperre aktiv. Bitte 5 Minuten warten.")
        else:
            st.error(f"Fehler bei {symbol_input}: {e}")

st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')} | Modus: {'🚀 Live'}")






