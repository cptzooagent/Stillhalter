import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
from curl_cffi import requests as crequests
from io import StringIO
from functools import partial

# --- GLOBALE SESSION ---
session = crequests.Session(impersonate="chrome")

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE & LOKALE TECHNIK ---

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def get_stock_data_from_batch(symbol, batch_df):
    try:
        hist = batch_df[symbol].dropna()
        if len(hist) < 30: return None
        close = hist['Close']
        price = close.iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi_val = 100 - (100 / (1 + (gain.iloc[-1] / (loss.iloc[-1] + 1e-9))))
        
        # Trend
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        is_uptrend = price > sma_200
        
        # Earnings aus Ticker-Objekt (muss leider online geladen werden)
        earn_str = "---"
        try:
            tk = yf.Ticker(symbol, session=session)
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass

        return price, rsi_val, is_uptrend, earn_str
    except: return None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', info.get('current_price', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER (+{rev_growth:.0f}%)", "#9b59b6"
        elif upside > 15: return f"✅ Stark (+{upside:.0f}%)", "#27ae60"
        return f"⚖️ Neutral (+{upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check", "#7f8c8d"

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = crequests.get(url, impersonate="chrome")
        df = pd.read_csv(StringIO(resp.text))
        tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
        extra = ["TSLA", "COIN", "MSTR", "PLTR", "HOOD", "SQ", "APP", "AVGO", "NVDA"]
        return list(set(tickers + extra))
    except: return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]

# --- 2. GLOBAL MARKET MONITORING ---

def get_market_data():
    try:
        m_tickers = ["^NDX", "^VIX", "BTC-USD"]
        m_data = yf.download(m_tickers, period="1mo", session=session, group_by='ticker', progress=False)
        ndx = m_data["^NDX"]
        cp_ndx = ndx['Close'].iloc[-1]
        sma20_ndx = ndx['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndx = ((cp_ndx - sma20_ndx) / sma20_ndx) * 100
        vix_val = m_data["^VIX"]['Close'].iloc[-1]
        btc_val = m_data["BTC-USD"]['Close'].iloc[-1]
        return cp_ndx, 50, dist_ndx, vix_val, btc_val
    except: return 0, 50, 0, 20, 0

st.markdown("## 🌍 Globales Markt-Monitoring")
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()

if vix_val > 25 or dist_ndq < -3:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Volatilität hoch"
elif vix_val < 15 and dist_ndq > 2:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Korrekturgefahr"
else:
    m_color, m_text = "#27ae60", "✅ TRENDSTARK: Konstruktives Umfeld"

st.markdown(f'<div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;"><h3>{m_text}</h3></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with c2: st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")
with c3: st.metric("Bitcoin", f"{btc_val:,.0f} $")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Strategie")
    otm_puffer_slider = st.slider("Puffer (%)", 3, 25, 15)
    min_yield_pa = st.number_input("Rendite p.a. (%)", 0, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis ($)", 0, 1000, (60, 500))
    st.markdown("---")
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 20)
    only_uptrend = st.checkbox("Nur SMA 200 Aufwärts", value=False)
    test_modus = st.checkbox("🛠️ Simulations-Modus", value=False)

# --- 4. PROFI-SCANNER ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", use_container_width=True):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    ticker_liste = ["APP", "NVDA", "TSLA", "PLTR", "COIN", "AVGO", "MSTR"] if test_modus else get_combined_watchlist()
    
    with st.spinner("🏁 Phase 1: Batch-Download Historie..."):
        batch_df = yf.download(ticker_liste, period="250d", session=session, group_by='ticker', threads=True, progress=False)

    with st.spinner("🔎 Phase 2: Detail-Analyse..."):
        pre_filtered = [s for s in ticker_liste if s in batch_df.columns.levels[0] and not batch_df[s].empty]
        all_results = []
        progress_bar = st.progress(0)

        def check_single_stock_optimized(symbol, b_df):
            try:
                # 1. LOKAL
                tech = get_stock_data_from_batch(symbol, b_df)
                if not tech: return None
                price, rsi, uptrend, earn_str = tech
                if not (min_stock_price <= price <= max_stock_price): return None
                if only_uptrend and not uptrend: return None

                # 2. ONLINE
                tk = yf.Ticker(symbol, session=session)
                m_cap = tk.fast_info.get("market_cap", 0)
                if m_cap < p_min_cap: return None

                dates = tk.options
                target_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                if not target_dates: return None
                
                chain = tk.option_chain(target_dates[0]).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]
                
                # Rendite & Risiko Metriken
                days = (datetime.strptime(target_dates[0], '%Y-%m-%d') - heute).days
                fair = (o['bid'] + o['ask']) / 2 if o['bid'] > 0 else o['lastPrice']
                y_pa = (fair / o['strike']) * (365 / max(1, days)) * 100
                if y_pa < p_min_yield: return None

                iv = o.get('impliedVolatility', 0.5)
                exp_move_pct = (iv * np.sqrt(days/365)) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 1.0
                delta_val = calculate_bsm_delta(price, o['strike'], days/365, iv)

                # Analysten
                info = tk.info
                analyst_txt, analyst_col = get_analyst_conviction(info)
                s_val = 2.0 if "Stark" in analyst_txt else 1.0
                if rsi < 35: s_val += 0.5
                if uptrend: s_val += 0.5

                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'rsi': rsi, 'bid': fair, 'delta': delta_val,
                    'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 'earn': earn_str,
                    'stars_str': "⭐" * int(s_val), 'analyst_label': analyst_txt, 
                    'analyst_color': analyst_col, 'mkt_cap': m_cap / 1e9,
                    'em_pct': exp_move_pct, 'em_safety': em_safety, 'sent_icon': "🟢" if uptrend else "🔵"
                }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            check_func = partial(check_single_stock_optimized, b_df=batch_df)
            futures = {executor.submit(check_func, s): s for s in pre_filtered}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: all_results.append(res)
                progress_bar.progress((i + 1) / len(pre_filtered))

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
        st.success(f"Gefunden: {len(all_results)} Trades")

# --- 5. ERGEBNIS-ANZEIGE (KACHELN) ---
if st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups ({len(all_results)} Treffer)")
    
    cols = st.columns(4)
    
    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            # Styling Logik vorbereiten
            rsi_val = int(res.get('rsi', 50))
            rsi_style = "color: #ef4444;" if rsi_val >= 70 else "color: #10b981;" if rsi_val <= 35 else "color: #4b5563;"
            em_safety = res.get('em_safety', 1.0)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            
            # WICHTIG: Der HTML-Block muss ganz links am Rand starten!
            html_code = f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
<span style="font-size: 1.1em; font-weight: 800; color: #111827;">{res['symbol']} <span style="font-size: 0.8em;">{res['stars_str']}</span></span>
<span style="font-size: 0.7em; font-weight: 700; color: #3b82f6; background: #3b82f610; padding: 2px 6px; border-radius: 4px;">{res['sent_icon']} {res['status']}</span>
</div>
<div style="margin: 8px 0;">
<div style="font-size: 0.6em; color: #6b7280; font-weight: 700; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 1.7em; font-weight: 900; color: #111827; line-height: 1.1;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
<div style="border-left: 2px solid #8b5cf6; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Strike</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['strike']:.1f}$</div>
</div>
<div style="border-left: 2px solid #f59e0b; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Mid</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['bid']:.2f}$</div>
</div>
</div>
<div style="background: {em_col}10; padding: 5px 8px; border-radius: 6px; margin-bottom: 10px; border: 1px dashed {em_col};">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.6em; color: #4b5563; font-weight: bold;">Safety: {em_safety:.1f}x EM</span>
<span style="font-size: 0.7em; font-weight: 800; color: {em_col};">±{res['em_pct']:.1f}%</span>
</div>
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 8px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.65em; color: #4b5563;">
<span>⏳ <b>{res['tage']}d</b></span>
<span style="background: #f3f4f6; padding: 1px 5px; border-radius: 3px; {rsi_style}">RSI: {rsi_val}</span>
<span style="font-weight: 800; color: #6b7280;">🗓️ {res['earn']}</span>
</div>
<div style="background: {res['analyst_color']}10; color: {res['analyst_color']}; padding: 6px; border-radius: 6px; margin-top: 8px; font-size: 0.65em; font-weight: 800;">
🚀 {res['analyst_label']}
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)
            
# --- SEKTION 2: DEPOT-MANAGER (MIT ALLEN WERTEN & SESSION-FIX) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

# Cache-Logik für schnellere Performance
if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell pausiert, um den Start zu beschleunigen.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
        with st.spinner("Berechne Pivot-Punkte und Signale..."):
            # Deine vollständige Asset-Liste
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
                "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    time.sleep(0.7) # Sicherheitspause für Yahoo
                    # 1. Daten über die stabile get_stock_data_full (mit Session) laden
                    res = get_stock_data_full(symbol)
                    if res is None or res[0] is None: continue
                    
                    price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # 2. KI-Analyse & Sterne-Logik (mit Session-Fix)
                    ki_status, _, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
                    
                    # Ticker-Abfrage für Sterne-Rating mit Session!
                    tk_temp = yf.Ticker(symbol, session=session)
                    info_temp = tk_temp.info
                    analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                    stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                    star_display = "⭐" * stars_count

                    # 3. Pivot-Daten extrahieren
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    r2_w = pivots.get('W_R2') if pivots else None
                    
                    # 4. Signal-Logik (Repair & Income)
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"
                    
                    call_action = "⏳ Warten"
                    if rsi > 55 and r2_d and price >= r2_d * 0.98: call_action = "🟢 JETZT (R2/RSI)"

                    # 5. Datensatz für die Tabelle
                    depot_list.append({
                        "Ticker": f"{symbol} {star_display}",
                        "Earnings": earn if earn else "---",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "KI-Check": f"{ki_icon} {ki_status}",
                        "RSI": int(rsi),
                        "Short Put (Repair)": put_action,
                        "Covered Call": call_action,
                        "S2 Daily": f"{s2_d:.2f} $" if s2_d else "---",
                        "S2 Weekly": f"{s2_w:.2f} $" if s2_w else "---",
                        "R2 Daily": f"{r2_d:.2f} $" if r2_d else "---",
                        "R2 Weekly": f"{r2_w:.2f} $" if r2_w else "---" 
                    })
                except Exception as e:
                    continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    # Header mit Refresh-Button
    col_header, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Daten aktualisieren"):
            st.session_state.depot_data_cache = None
            st.rerun()

    # Anzeige der vollständigen Tabelle
    st.table(pd.DataFrame(st.session_state.depot_data_cache))

# --- SEKTION 3: PROFI-ANALYSE & TRADING-COCKPIT ---
st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")

# Wir nutzen ein Formular, damit nicht jeder Tastendruck einen Re-Run auslöst
with st.form("cockpit_form"):
    symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()
    submit_button = st.form_submit_button("🚀 Analyse starten / aktualisieren")

if submit_button and symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            # Hier nutzen wir jetzt die globale Session! 
            tk = yf.Ticker(symbol_input, session=session) 
            info = tk.info
            res = get_stock_data_full(symbol_input)

            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
                analyst_txt, analyst_col = get_analyst_conviction(info)

                # --- Earnings-Anzeige ---
                if earn and earn != "---":
                    # Dynamische Prüfung für 2026 [cite: 40]
                    st.info(f"🗓️ Nächste Earnings: {earn}")
                else:
                    st.write("🗓️ Keine Earnings-Daten verfügbar")

                # --- STRATEGIE-SIGNAL ---
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
                
                put_action_scanner = "⏳ Warten (Kein Signal)"
                signal_color = "white"

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b"
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60"

                st.markdown(f"""
                    <div style="padding:10px; border-radius:10px; border: 2px solid {signal_color}; text-align:center;">
                        <small>Aktuelles Short Put Signal:</small><br>
                        <strong style="font-size:20px; color:{signal_color};">{put_action_scanner}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                
                # Sterne-Logik
                stars = 0
                if "HYPER" in analyst_txt: stars = 3
                elif "Stark" in analyst_txt: stars = 2
                elif "Neutral" in analyst_txt: stars = 1
                if uptrend and stars > 0: stars += 0.5
                
                # --- VERSCHÄRFTE AMPEL-LOGIK ---
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif stars >= 2.5 and uptrend and 30 <= rsi <= 60:
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ANALYSTEN-WARNUNG"

                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="margin:0; font-size: 1.8em; letter-spacing: 1px;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Kurs", f"{price:.2f} $")
                with col2: st.metric("RSI (14)", f"{int(rsi)}", delta="PANIK" if rsi < 25 else None, delta_color="inverse")
                with col3: 
                    status_icon = "🛡️" if uptrend else "💎"
                    st.metric("Phase", f"{status_icon} {'Trend' if uptrend else 'Dip'}")
                with col4: st.metric("Qualität", "⭐" * int(stars))

                # --- PIVOT ANALYSE ---
                st.markdown("---")
                if pivots_res:
                    st.markdown("#### 🛡️ Technische Absicherung & Ziele (Pivots)")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Weekly S2 (Boden)", f"{pivots_res['W_S2']:.2f} $")
                    pc2.metric("Daily S2", f"{pivots_res['S2']:.2f} $")
                    pc3.metric("Pivot (P)", f"{pivots_res['P']:.2f} $")
                    pc4.metric("Daily R2 (Ziel)", f"{pivots_res['R2']:.2f} $")
                    pc5.metric("Weekly R2 (Top)", f"{pivots_res['W_R2']:.2f} $")
                    st.caption(f"💡 **CC-Tipp:** Ein Covered Call am R2 Weekly ({pivots_res['W_R2']:.2f} $) bietet statistische Sicherheit.")

                # Analysten Box
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                    </div>
                """, unsafe_allow_html=True)

                # --- OPTIONEN TABELLE & UMSCHALTER ---
                st.markdown("---")
                st.markdown("### 🎯 Option-Chain Auswahl")
                option_mode = st.radio("Strategie wählen:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                heute = datetime.now()
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    days_to_expiry = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - heute).days)

                    # OpenClaw KI-Box
                    ki_status, ki_text, ki_score = get_openclaw_analysis(symbol_input)
                    st.info(ki_text)

                    opt_chain = tk.option_chain(target_date)
                    chain = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    df_disp = chain[chain['openInterest'] > 50].copy()

                    if "Put" in option_mode:
                        df_disp = df_disp[df_disp['strike'] < price].copy()
                        df_disp['Puffer %'] = ((price - df_disp['strike']) / price) * 100
                        sort_order = False
                    else:
                        df_disp = df_disp[df_disp['strike'] > price].copy()
                        df_disp['Puffer %'] = ((df_disp['strike'] - price) / price) * 100
                        sort_order = True

                    df_disp['Yield p.a. %'] = (df_disp['bid'] / df_disp['strike']) * (365 / days_to_expiry) * 100
                    df_disp = df_disp.sort_values('strike', ascending=sort_order)

                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 5 <= p < 10: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    styled_df = df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(15).style.apply(style_rows, axis=1).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $', 'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                    })
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    st.caption("🟢 >10% Puffer | 🟡 5-10% Puffer | 🔴 <5% Puffer (Risiko)")

    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live-Scan'}")



