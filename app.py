import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
from curl_cffi import requests as crequests

# Globale Session für Stabilität
session = crequests.Session(impersonate="chrome")

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro-Scanner", layout="wide")

# --- 1. DATEN-STRUKTUR (DIE TOP 200) ---
tickers_dict = {
    "BIG_TECH": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "NFLX", "ADBE", "AVGO", "ORCL", "CRM", "INTU", "CSCO", "ASML", "AMD", "QCOM", "TXN", "AMAT", "MU"],
    "SOFTWARE_GROWTH": ["PLTR", "SNOW", "PANW", "SNPS", "CDNS", "WDAY", "ROP", "TEAM", "DDOG", "NET", "OKTA", "ZS", "SHOP", "SPOT", "UBER", "ABNB", "SQ", "PYPL", "COIN", "MSTR"],
    "CHIPS_HARDWARE": ["LRCX", "ADI", "KLAC", "NXPI", "MRVL", "MCHP", "MPWR", "ON", "INTC", "ARM", "SMCI", "STX", "WDC", "HPQ", "DELL", "HPE", "VRT", "ANET", "TEL", "APH"],
    "FINANZ_BANKEN": ["JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "PYPL", "COF", "WFC", "C", "USB", "PNC", "TFC", "BK", "STT", "SCHW", "IBKR", "BRK-B"],
    "KONSUM_RETAIL": ["WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "EL", "CL", "PG", "KO", "PEP", "MDLZ", "PM", "MO", "TJX", "DG", "DLTR", "ROST"],
    "PHARMA_HEALTH": ["LLY", "JNJ", "PFE", "MRK", "ABBV", "AMGN", "UNH", "CVS", "ISRG", "GILD", "VRTX", "REGN", "BSX", "SYK", "ZTS", "BDX", "MCK", "ABC", "CI", "HUM"],
    "ENERGIE_INDUSTRIE": ["XOM", "CVX", "SLB", "COP", "EOG", "PXD", "MPC", "PSX", "CAT", "DE", "BA", "LMT", "GD", "NOC", "RTX", "HON", "GE", "MMM", "UPS", "FDX"],
    "COMMUNICATION_UTILITIES": ["VZ", "T", "TMUS", "CMCSA", "DIS", "WBD", "PARA", "CHTR", "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "FE", "ETR", "PEG", "ED"],
    "ETFS_INDICES": ["SPY", "QQQ", "IWM", "DIA", "XBI", "GDX", "GLD", "SLV", "TLT", "EEM", "ARKK", "SMH", "XLF", "XLV", "XLE", "XLI", "XLK", "XLU", "XLY", "XLP"],
    "SPECIAL_WATCH": ["AFRM", "HOOD", "RIVN", "LCID", "DKNG", "PINS", "SNAP", "ZM", "DOCU", "ROKU", "AI", "SOFI", "UPST", "MARA", "RIOT", "U", "BABA", "PDD", "JD", "BIDU"]
}

# --- 2. HILFSFUNKTIONEN ---
def get_cnn_fear_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36'}
        r = session.get(url, headers=headers, timeout=5)
        return int(r.json()['fear_and_greed']['score'])
    except: return 50

def get_market_data():
    try:
        data = yf.download(["^IXIC", "^VIX"], period="6mo", interval="1d", progress=False)
        ndq = data['Close']['^IXIC'].dropna()
        cp_ndq = ndq.iloc[-1]
        vix_val = data['Close']['^VIX'].iloc[-1]
        
        # RSI
        delta = ndq.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_ndq = 100 - (100 / (1 + (gain / loss).iloc[-1]))
        dist_ndq = ((cp_ndq - ndq.rolling(20).mean().iloc[-1]) / ndq.rolling(20).mean().iloc[-1]) * 100
        
        return cp_ndq, rsi_ndq, dist_ndq, vix_val, get_cnn_fear_greed()
    except: return 17000, 50, 0, 15, 50

# --- SIDEBAR (DIE STEUERZENTRALE) ---
with st.sidebar:
    st.header("⚙️ Scanner Settings")
    
    # NEU: Test-Modus Toggle
    test_modus = st.toggle("🛠️ Test-Modus (Schnellscan)", value=True, help="Scannt nur eine kleine Auswahl für schnelles Testing.")
    
    # Sektoren-Auswahl
    selected_sectors = st.multiselect(
        "Sektoren wählen", 
        options=list(tickers_dict.keys()), 
        default=["BIG_TECH", "ETFS_INDICES"]
    )
    
    st.divider()
    
    # Stillhalter Parameter
    st.subheader("Stillhalter Parameter")
    target_delta = st.slider("Ziel Delta (Put)", 0.05, 0.30, 0.15, 0.01)
    min_yield = st.slider("Min. Rendite p.a. %", 5, 50, 15)
    dte_range = st.slider("Laufzeit (DTE)", 7, 60, (14, 45))
    
    st.divider()
    scan_button = st.button("🚀 SCAN STARTEN", use_container_width=True)

# --- TICKER LOGIK ANPASSEN ---
final_tickers = []
for sector in selected_sectors:
    final_tickers.extend(tickers_dict[sector])

# Wenn Test-Modus aktiv, Liste kürzen, damit es keine Timeouts gibt
if test_modus:
    final_tickers = final_tickers[:15] # Nur die ersten 15 Aktien zum Testen

# --- MAIN UI ---
cp_ndq, rsi_ndq, dist_ndq, vix_val, stock_fg = get_market_data()

st.title("🦅 Stillhalter Pro-Scanner")

# Der Trendbalken
m_color = "#e74c3c" if dist_ndq < -2 or vix_val > 25 or stock_fg < 30 else "#f39c12" if rsi_ndq > 72 or stock_fg > 78 else "#27ae60"
m_text = "🚨 MARKT-ALARM" if m_color == "#e74c3c" else "⚠️ ÜBERHITZT" if m_color == "#f39c12" else "✅ TRENDSTARK"

st.markdown(f"""
    <div style="background: linear-gradient(90deg, {m_color}dd, {m_color}); 
                color: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
        <h2 style="margin:0;">{m_text}</h2>
        <p style="margin:5px 0 0 0; opacity: 0.9;">Basis für Stillhalter-Entscheidungen</p>
    </div>
    """, unsafe_allow_html=True)

# KPI Kacheln
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
c2.metric("VIX (Angst)", f"{vix_val:.2f}", delta="Erhöht" if vix_val > 20 else "Niedrig", delta_color="inverse")
c3.metric("Fear & Greed", f"{stock_fg}", "CNN Business")
c4.metric("Nasdaq RSI", f"{int(rsi_ndq)}", "Überkauft" if rsi_ndq > 70 else "Neutral")

st.divider()

# Vorbereitung der Ticker-Liste basierend auf Sidebar-Wahl
final_tickers = []
for sector in selected_sectors:
    final_tickers.extend(tickers_dict[sector])


# --- HIER STARTET DEIN PROFI-SCANNER ---

# --- SEKTION 1: PROFI-SCANNER (TURBO-HYBRID-VERSION) ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro", use_container_width=True):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("🏁 Phase 1: Batch-Download & Vor-Filterung..."):
        # Ticker-Liste bestimmen
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
        
        # 1. BATCH-DOWNLOAD: Alle Kurse in einem Rutsch laden (spart hunderte Einzel-Requests)
        try:
            batch_df = yf.download(ticker_liste, period="150d", session=session, group_by='ticker', threads=True, progress=False)
        except Exception as e:
            st.error(f"Batch-Download fehlgeschlagen: {e}")
            batch_df = pd.DataFrame()

        # 2. VOR-FILTERUNG: Wer passt überhaupt ins Raster?
        filtered_tickers = []
        for symbol in ticker_liste:
            try:
                # Prüfen, ob Daten für den Ticker im Batch vorhanden sind
                if symbol not in batch_df.columns.levels[0]: continue
                s_hist = batch_df[symbol]
                if s_hist.empty or len(s_hist) < 2: continue
                
                last_price = s_hist['Close'].iloc[-1]
                
                # Schneller Preis-Check
                if not (min_stock_price <= last_price <= max_stock_price):
                    continue
                
                # Wenn Preis passt, ab in die Detail-Analyse
                filtered_tickers.append(symbol)
            except:
                continue

    if not filtered_tickers:
        st.warning("Keine Aktien gefunden, die den Preis-Filter erfüllen.")
    else:
        st.info(f"🔎 Phase 2: Detail-Analyse für {len(filtered_tickers)} Top-Kandidaten...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []

        # Interne Funktion für Multithreading
        def check_single_stock_optimized(symbol):
            try:
                # 0. Kurze Pause für API-Stabilität
                time.sleep(0.5)
                tk = yf.Ticker(symbol, session=session)
                
                # 1. FAST_INFO für Market Cap & Preis-Validierung
                fast = tk.fast_info
                price = fast.get("last_price") or fast.get("lastPrice")
                m_cap = fast.get("market_cap") or fast.get("marketCap")
                if not price or m_cap < p_min_cap: return None

                # 2. TECHNIK-CHECK (RSI & Trend)
                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: return None
                _, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                if only_uptrend and not uptrend: return None

                # 3. OPTIONS-CHECK
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 30]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]
                
                # Rendite-Check
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                bid, ask = o['bid'], o['ask']
                fair_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else o['lastPrice']
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                
                if y_pa < p_min_yield: return None

                # 4. QUALITÄTS-CHECK (Sterne-Rating)
                info = tk.info
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                s_val = 3.0 if "HYPER" in analyst_txt else 2.0 if "Stark" in analyst_txt else 1.0 if "Neutral" in analyst_txt else 0.0
                if rsi < 35: s_val += 0.5
                if uptrend: s_val += 0.5
                
                # Risiko-Metriken
                iv = o.get('impliedVolatility', 0.4)
                exp_move_pct = (price * (iv * np.sqrt(days_to_exp / 365)) / price) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)

                return {
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'bid': fair_price, 'rsi': rsi, 'earn': earn if earn else "---", 
                    'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 'delta': delta_val,
                    'stars_val': s_val, 'stars_str': "⭐" * int(s_val) if s_val >= 1 else "⚠️",
                    'analyst_label': analyst_txt, 'analyst_color': analyst_col, 'mkt_cap': m_cap / 1e9,
                    'em_pct': exp_move_pct, 'em_safety': em_safety
                }
            except:
                return None

        # PARALLELE AUSFÜHRUNG (10 Workers für maximalen Speed)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(check_single_stock_optimized, s): s for s in filtered_tickers}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(filtered_tickers))
                if i % 10 == 0: status_text.text(f"Analysiere Kandidat {i}/{len(filtered_tickers)}...")
        
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            st.session_state.profi_scan_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)
            st.success(f"✅ Scan beendet: {len(all_results)} profitable Trades gefunden!")
        else:
            st.warning("Keine Treffer mit den aktuellen Filtern (Puffer/Rendite) gefunden.")

# --- KORRIGIERTER ANZEIGEBLOCK (HTML-FIX) ---
if 'profi_scan_results' in st.session_state and st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(all_results)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    
    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            # Variablen-Vorbereitung [cite: 160-167]
            earn_str = res.get('earn', "---")
            status_txt = res.get('status', "Trend")
            sent_icon = res.get('sent_icon', "🟢")
            stars = res.get('stars_str', "⭐")
            s_color = "#10b981" if "Trend" in status_txt else "#3b82f6"
            a_label = res.get('analyst_label', "Keine Analyse")
            a_color = res.get('analyst_color', "#8b5cf6")
            mkt_cap = res.get('mkt_cap', 0)
            rsi_val = int(res.get('rsi', 50))
            
            # RSI & Delta Styling [cite: 165, 166]
            rsi_style = "color: #ef4444;" if rsi_val >= 70 else "color: #10b981;" if rsi_val <= 35 else "color: #4b5563;"
            delta_val = abs(res.get('delta', 0))
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            
            # EM Sicherheit [cite: 151, 167]
            em_safety = res.get('em_safety', 1.0)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            
            # Earnings Risiko [cite: 168]
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    # Fix für Datumsformat (Jahr 2026 ergänzt)
                    earn_date = datetime.strptime(f"{earn_str}2026", "%d.%m.%Y")
                    if 0 <= (earn_date - heute_dt).days <= res.get('tage', 14): 
                        is_earning_risk = True
                except: pass
            
            card_border, card_shadow, card_bg = ("3px solid #ef4444", "0 4px 12px rgba(239,68,68,0.2)", "#fffcfc") if is_earning_risk else ("1px solid #e5e7eb", "0 2px 4px rgba(0,0,0,0.05)", "#ffffff")

            # Das f-String HTML muss ohne zusätzliche Tabs am Zeilenanfang stehen, damit Streamlit es sauber liest
            html_code = f"""<div style="background: {card_bg}; border: {card_border}; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: {card_shadow}; font-family: sans-serif; text-align: left;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
<span style="font-size: 1.1em; font-weight: 800; color: #111827;">{res['symbol']} <span style="font-size: 0.8em;">{stars}</span></span>
<span style="font-size: 0.7em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 2px 6px; border-radius: 4px;">{sent_icon} {status_txt}</span>
</div>
<div style="margin: 8px 0; text-align: left;">
<div style="font-size: 0.6em; color: #6b7280; font-weight: 700; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 1.7em; font-weight: 900; color: #111827; line-height: 1.1;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; text-align: left;">
<div style="border-left: 2px solid #8b5cf6; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Strike</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['strike']:.1f}$</div>
</div>
<div style="border-left: 2px solid #f59e0b; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Mid</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['bid']:.2f}$</div>
</div>
<div style="border-left: 2px solid #3b82f6; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Puffer</div>
<div style="font-size: 0.85em; font-weight: 700;">{res['puffer']:.1f}%</div>
</div>
<div style="border-left: 2px solid {delta_col}; padding-left: 6px;">
<div style="font-size: 0.55em; color: #6b7280;">Delta</div>
<div style="font-size: 0.85em; font-weight: 700; color: {delta_col};">{delta_val:.2f}</div>
</div>
</div>
<div style="background: {em_col}10; padding: 5px 8px; border-radius: 6px; margin-bottom: 10px; border: 1px dashed {em_col}; text-align: left;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.6em; color: #4b5563; font-weight: bold;">Stat. Erwartung:</span>
<span style="font-size: 0.7em; font-weight: 800; color: {em_col};">±{res['em_pct']:.1f}%</span>
</div>
<div style="font-size: 0.55em; color: #6b7280;">Sicherheit: <b>{em_safety:.1f}x EM</b></div>
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 8px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.65em; color: #4b5563; margin-bottom: 8px;">
<span>⏳ <b>{res['tage']}d</b></span>
<span style="background: #f3f4f6; padding: 1px 5px; border-radius: 3px; {rsi_style} font-weight: 700;">RSI: {rsi_val}</span>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">{'⚠️' if is_earning_risk else '🗓️'} {earn_str}</span>
</div>
<div style="background: {a_color}10; color: {a_color}; padding: 6px; border-radius: 6px; border-left: 3px solid {a_color}; font-size: 0.65em; font-weight: 800; text-align: left;">
🚀 {a_label}
</div>
</div>"""
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




