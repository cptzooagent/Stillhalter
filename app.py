import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time

# --- 1. REINIGUNG & INITIALISIERUNG (Fix für curl_cffi & _expire_after) ---
# Wir lassen yfinance die Verbindung selbst verwalten, um TLS-Konflikte zu vermeiden.

def get_tk(symbol):
    """Gibt ein Standard Ticker-Objekt zurück. 
    yfinance nutzt intern curl_cffi, wenn es in den Requirements steht."""
    return yf.Ticker(symbol)

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE & TECHNIK ---
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

def calculate_pivots(symbol):
    """Berechnet Daily und Weekly Pivot-Punkte."""
    try:
        tk = get_tk(symbol)
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s1_d = (2 * p_d) - h_d
        s2_d = p_d - (h_d - l_d)
        r2_d = p_d + (h_d - l_d) 

        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: 
            return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_d, "W_R2": r2_d}
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        s2_w = p_w - (h_w - l_w)
        r2_w = p_w + (h_w - l_w) 

        return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_w, "W_R2": r2_w}
    except: return None

def get_openclaw_analysis(symbol):
    try:
        tk = get_tk(symbol)
        all_news = tk.news
        if not all_news or len(all_news) == 0:
            return "Neutral", "🤖 OpenClaw: Yahoo liefert aktuell keine News.", 0.5
        huge_blob = str(all_news).lower()
        display_text = all_news[0].get('title', 'Marktstimmung aktiv')
        
        score = 0.5
        bull_words = ['earnings', 'growth', 'beat', 'buy', 'profit', 'ai', 'demand', 'up', 'bull', 'upgrade']
        bear_words = ['sell-off', 'disruption', 'miss', 'down', 'risk', 'decline', 'short', 'warning', 'sell']
        for w in bull_words:
            if w in huge_blob: score += 0.08
        for w in bear_words:
            if w in huge_blob: score -= 0.08
        score = max(0.1, min(0.9, score))
        status = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
        icon = "🟢" if status == "Bullish" else "🔴" if status == "Bearish" else "🟡"
        return status, f"{icon} OpenClaw: {display_text[:90]}", score
    except: return "N/A", "🤖 OpenClaw: System-Standby...", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except: return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

def get_stock_data_full(symbol):
    try:
        tk = get_tk(symbol)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0, None
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        pivots = calculate_pivots(symbol)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, is_uptrend, False, atr, pivots
    except: return None, [], "", 50, True, False, 0, None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}%)", "#9b59b6"
        elif upside > 15: return f"✅ Stark (+{upside:.0f}%)", "#27ae60"
        elif upside < 0: return f"⚠️ Warnung ({upside:.1f}%)", "#e67e22"
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
def get_market_data():
    try:
        # Kurze Pause für API-Stabilität
        time.sleep(0.5)
        ndq = get_tk("^NDX"); vix = get_tk("^VIX")
        h_ndq = ndq.history(period="1mo"); h_vix = vix.history(period="1d")
        if h_ndq.empty: return 0, 50, 0, 20
        cp_ndq = h_ndq['Close'].iloc[-1]
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        v_val = h_vix['Close'].iloc[-1] if not h_vix.empty else 20
        return cp_ndq, 50, dist_ndq, v_val
    except: return 0, 50, 0, 20

st.markdown("## 🌍 Globales Markt-Monitoring")
cp_ndq, rsi_ndq, dist_ndq, vix_val = get_market_data()

if dist_ndq < -2 or vix_val > 25:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Volatilität erhöht"
else:
    m_color, m_text = "#27ae60", "✅ STABIL: Umfeld für Stillhalter geeignet"

st.markdown(f'''
    <div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0;">{m_text}</h3>
    </div>
''', unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}%")
col_m2.metric("VIX (Angst)", f"{vix_val:.2f}")
col_m3.metric("Status", "Risk-On" if vix_val < 20 else "Risk-Off")
st.markdown("---")

# --- SEKTION 2: PROFI-SCANNER (STABILISIERT & DESIGN-FIX) ---

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Scanner analysiert Ticker... (Drosselung gegen Sperren aktiv)"):
        ticker_liste = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "COIN", "PLTR", "MU", "CRWD", "MARA", "AFRM", "HIMS", "NET", "AVGO", "MSTR"] if test_modus else get_combined_watchlist()
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        all_results = []

        def check_single_stock(symbol):
            try:
                # Wichtig: 1.5s Pause verhindert 'Too Many Requests' (Bild 1)
                time.sleep(1.5) 
                tk = get_tk(symbol)
                info = tk.info
                price = info.get('currentPrice', info.get('regularMarketPrice'))
                if not price or not (min_stock_price <= price <= max_stock_price): return None
                
                res = get_stock_data_full(symbol)
                if not res or res[0] is None: return None
                price, dates, earn, rsi, uptrend, _, atr, pivots = res
                
                if only_uptrend and not uptrend: return None
                
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                
                opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] >= 0)].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]

                bid = o['bid'] if o['bid'] > 0 else (o['lastPrice'] * 0.95)
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (bid / o['strike']) * (365 / max(1, days_to_exp)) * 100
                
                if y_pa < p_min_yield: return None

                iv = o.get('impliedVolatility', 0.5)
                exp_move_pct = (iv * np.sqrt(days_to_exp / 365)) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 1.0
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                
                analyst_txt, analyst_col = get_analyst_conviction(info)
                ki_status, ki_text, _ = get_openclaw_analysis(symbol)
                
                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'bid': bid, 'rsi': rsi, 'earn': earn, 
                    'tage': days_to_exp, 'stars': "⭐" * 3 if "HYPER" in analyst_txt else "⭐" * 2, 
                    'delta': abs(delta_val), 'analyst_label': analyst_txt, 'analyst_color': analyst_col,
                    'ki_info': ki_text, 'uptrend': uptrend, 'em_pct': exp_move_pct, 'em_safety': em_safety,
                    'm_cap': info.get('marketCap', 0) / 1e9
                }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste[:40]}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(futures))

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
        st.rerun()

# --- ANZEIGE DER KACHELN (DESIGN FIX: LINKSBÜNDIG FÜR STREAMLIT RENDERING) ---
if st.session_state.profi_scan_results:
    res_list = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(res_list)} Treffer)")
    cols = st.columns(4)
    
    for idx, res in enumerate(res_list):
        with cols[idx % 4]:
            trend_icon = "🟢" if res['uptrend'] else "🟡"
            trend_label = "Trend" if res['uptrend'] else "Dip"
            
            # WICHTIG: Das HTML klebt am linken Rand!
            html_card = f"""
<div style="border: 1px solid #e0e0e0; border-radius: 15px; padding: 16px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif; height: 530px; position: relative; margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div><b style="font-size: 1.4em; color: #111827;">{res['symbol']}</b><span style="margin-left: 5px;">⚠️</span></div>
<div style="background: #f0fdf4; border: 1px solid #dcfce7; padding: 2px 8px; border-radius: 10px; font-size: 0.65em; display: flex; align-items: center;">
{trend_icon} <span style="color: #166534; margin-left: 4px; font-weight: bold;">🛡️ {trend_label}</span>
</div>
</div>
<p style="margin: 10px 0 0 0; font-size: 0.65em; color: #6b7280; letter-spacing: 0.05em;">YIELD P.A.</p>
<div style="font-size: 2.2em; font-weight: 800; color: #111827; margin-bottom: 15px;">{res['y_pa']:.1f}%</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;"><span style="font-size: 0.6em; color: #6b7280;">Strike</span><br><b>{res['strike']:.1f}$</b></div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;"><span style="font-size: 0.6em; color: #6b7280;">Mid</span><br><b>{res['bid']:.2f}$</b></div>
<div style="border-left: 3px solid #3b82f6; padding-left: 8px;"><span style="font-size: 0.6em; color: #6b7280;">Puffer</span><br><b>{res['puffer']:.1f}%</b></div>
<div style="border-left: 3px solid #10b981; padding-left: 8px;"><span style="font-size: 0.6em; color: #6b7280;">Delta</span><br><b style="color: #10b981;">{res['delta']:.2f}</b></div>
</div>
<div style="border: 1px dashed #f59e0b; border-radius: 8px; padding: 8px; background: #fffcf0; margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; font-size: 0.7em; font-weight: bold;">
<span style="color: #4b5563;">Stat. Erwartung (EM):</span><span style="color: #f59e0b;">±{res['em_pct']:.1f}%</span>
</div>
<p style="margin: 4px 0 0 0; font-size: 0.65em; color: #6b7280;">Sicherheit: {res['em_safety']:.1f}x EM</p>
</div>
<div style="background: #f3f4f6; border-radius: 8px; padding: 8px; font-size: 0.65em; color: #374151; display: flex; align-items: center; margin-bottom: 12px; border-left: 4px solid #9ca3af;">
<span style="margin-right: 6px;">🟡</span> {res['ki_info'][:65]}...
</div>
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.7em; color: #4b5563; margin-bottom: 12px;">
<span>⏳ {res['tage']}d</span>
<span style="background: #f0fdf4; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: bold;">RSI: {int(res['rsi'])}</span>
<span>{res['m_cap']:.0f}B</span>
<span>📅 {res['earn']}</span>
</div>
<div style="background: #fff7ed; color: {res['analyst_color']}; border: 1px solid #ffedd5; padding: 6px; border-radius: 8px; font-size: 0.65em; text-align: center; font-weight: bold;">
🚀 {res['analyst_label']}
</div>
</div>
"""
            st.markdown(html_card, unsafe_allow_html=True)
                    
# --- SEKTION 2: DEPOT-MANAGER (STABILISIERTE VERSION) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell pausiert, um den Start zu beschleunigen.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
        with st.spinner("Berechne Pivot-Punkte und Signale (gedrosselt)..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
                "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    # Sicherheits-Pause für Yahoo
                    time.sleep(1.2) 
                    res = get_stock_data_full(symbol)
                    if res is None or res[0] is None: continue
                    
                    price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # Sterne & KI (Nutzen intern get_tk)
                    ki_status, _, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
                    
                    # Analysten-Check via info (Session-basiert)
                    info_temp = get_tk(symbol).info
                    analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                    stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                    star_display = "⭐" * stars_count

                    # Pivot Logik
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"
                    
                    call_action = "⏳ Warten"
                    if rsi > 55 and r2_d and price >= r2_d * 0.98: call_action = "🟢 JETZT (R2/RSI)"

                    depot_list.append({
                        "Ticker": f"{symbol} {star_display}",
                        "Earnings": earn if earn else "---",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "KI-Check": f"{ki_icon} {ki_status}",
                        "RSI": int(rsi),
                        "Short Put (Repair)": put_action,
                        "Covered Call": call_action
                    })
                except:
                    continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    col_header, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Daten aktualisieren"):
            st.session_state.depot_data_cache = None
            st.rerun()
    st.table(pd.DataFrame(st.session_state.depot_data_cache))

# --- SEKTION 3: PROFI-ANALYSE & TRADING-COCKPIT ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU").upper()

if symbol_input:
    try:
        with st.spinner(f"Lade Daten für {symbol_input}..."):
            tk = get_tk(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)

            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
                analyst_txt, analyst_col = get_analyst_conviction(info)

                # Earnings-Check (2026 dynamisch)
                if earn and earn != "---":
                    st.info(f"🗓️ Nächste Earnings: {earn}")
                
                # Signal-Ampel
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
                
                put_action_scanner = "⏳ Warten"
                signal_color = "#f1c40f" # Gelb

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b"
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60"

                st.markdown(f"""
                    <div style="background-color: {signal_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <h2 style="margin:0;">{put_action_scanner}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Option Chain
                st.markdown("---")
                option_mode = st.radio("Strategie:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 45]
                if valid_dates:
                    target_date = st.selectbox("Verfallstag", valid_dates)
                    opt_chain = tk.option_chain(target_date)
                    df_disp = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    
                    # Filter & Berechnung
                    df_disp = df_disp[df_disp['openInterest'] > 10].copy()
                    if "Put" in option_mode:
                        df_disp = df_disp[df_disp['strike'] < price]
                        sort_asc = False
                    else:
                        df_disp = df_disp[df_disp['strike'] > price]
                        sort_asc = True
                    
                    df_disp['Yield p.a. %'] = (df_disp['bid'] / df_disp['strike']) * (365 / 30) * 100 # Vereinfacht
                    st.dataframe(df_disp[['strike', 'bid', 'ask', 'openInterest', 'Yield p.a. %']].sort_values('strike', ascending=sort_asc).head(10))

    except Exception as e:
        if "Too Many Requests" in str(e):
            st.error("🚫 Yahoo-Sperre aktiv. Bitte 5 Minuten warten oder VPN wechseln.")
        else:
            st.error(f"Fehler: {e}")

st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')} | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live'}")






