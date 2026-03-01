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

# Globale curl_cffi Session für yfinance
session = crequests.Session(impersonate="chrome")

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
    try:
        tk = yf.Ticker(symbol, session=session)
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
        tk = yf.Ticker(symbol, session=session)
        all_news = tk.news
        if not all_news:
            return "Neutral", "🤖 OpenClaw: Yahoo liefert aktuell keine Daten.", 0.5
        huge_blob = str(all_news).lower()
        display_text = ""
        for n in all_news:
            for val in n.values():
                if isinstance(val, str) and val.count(" ") > 3:
                    display_text = val
                    break
            if display_text: break
        if not display_text:
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
    except Exception: return "N/A", "🤖 OpenClaw: System-Reset...", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    # 1. Die Sektor-ETFs (Deine neue ETF-Spalte)
    sector_etfs = ["XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "XBI", "GDX", "ARKK", "SPY", "QQQ", "IWM"]
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = crequests.get(url, impersonate="chrome")
        df = pd.read_csv(StringIO(resp.text))
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        sp500_list = list(set(tickers + nasdaq_extra))
        sp500_list = [t.replace('.', '-') for t in sp500_list]
        return sp500_list, sector_etfs
    except: 
        backup = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]
        return backup, sector_etfs

def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol, session=session)
        hist = tk.history(period="150d")
        if hist.empty: return None, [], "", 50, True, False, 0, None
        price = hist['Close'].iloc[-1]
        dates = list(tk.options)
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        is_uptrend = price > sma_200
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        pivots = calculate_pivots(symbol)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr, pivots
    except: return None, [], "", 50, True, False, 0, None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', info.get('current_price', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        elif upside > 15 and rev_growth > 5: return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        elif upside > 25: return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        elif upside < 0 or rev_growth < -2: return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

# --- 1.1 DATEN LADEN (VOR DER SIDEBAR) ---
sp500_tickers, etf_tickers = get_combined_watchlist()

# --- 2. UI & SIDEBAR (KOMPAKT-VERSION) ---
with st.sidebar:
    st.title("🛡️ Scanner-Setup")
    
    # Simulation & Markt-Auswahl direkt oben für schnellen Zugriff
    test_modus = st.toggle("🛠️ Simulations-Modus", value=False, key="sim_checkbox")
    
    scan_mode = st.selectbox(
        "🎯 Markt-Auswahl",
        ["S&P 500 & Tech", "Nur Sektor-ETFs", "Alles kombinieren"],
        key="scan_mode_selection"
    )

    st.markdown("---")

    # Expander für Strategie-Details spart massiv vertikalen Platz
    with st.expander("📊 Strategie-Parameter", expanded=True):
        otm_puffer_slider = st.number_input("Puffer (%)", 3, 25, 15, key="puffer_sid")
        min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12, key="yield_sid")
        
        # Preisspanne in zwei kompakten Spalten nebeneinander
        st.write("Aktienpreis ($)")
        c1, c2 = st.columns(2)
        with c1: 
            min_stock_price = st.number_input("Min", 0, 1000, 60, key="p_min")
        with c2: 
            max_stock_price = st.number_input("Max", 0, 1000, 500, key="p_max")

    # Zweiter Expander für die eher statischen Filter
    with st.expander("🔍 Qualitäts-Filter", expanded=False):
        min_mkt_cap = st.select_slider(
            "Min. Cap (Mrd. $)", 
            options=[1, 5, 10, 20, 50, 100, 500, 1000], 
            value=20, 
            key="mkt_cap_sid"
        )
        only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")

# --- 3. TICKER LOGIK ANWENDEN ---
if test_modus:
    ticker_liste = ["NVDA", "TSLA", "AAPL", "PLTR", "XLE", "ARKK", "XBI"]
else:
    if scan_mode == "Nur Sektor-ETFs":
        ticker_liste = etf_tickers
    elif scan_mode == "Alles kombinieren":
        ticker_liste = list(set(sp500_tickers + etf_tickers))
    else:
        ticker_liste = sp500_tickers

def get_market_data():
    try:
        ndq = yf.Ticker("^NDX", session=session); vix = yf.Ticker("^VIX", session=session); btc = yf.Ticker("BTC-USD", session=session)
        h_ndq = ndq.history(period="1mo"); h_vix = vix.history(period="1d"); h_btc = btc.history(period="1d")
        if h_ndq.empty: return 0, 50, 0, 20, 0
        cp_ndq = h_ndq['Close'].iloc[-1]
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        delta = h_ndq['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_ndq = 100 - (100 / (1 + rs)).iloc[-1]
        v_val = h_vix['Close'].iloc[-1] if not h_vix.empty else 20
        b_val = h_btc['Close'].iloc[-1] if not h_btc.empty else 0
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except: return 0, 50, 0, 20, 0

def get_crypto_fg():
    try:
        r = crequests.get("https://api.alternative.me/fng/", impersonate="chrome")
        return int(r.json()['data'][0]['value'])
    except: return 50

# NEU: CNN Fear & Greed Scraper
def get_cnn_fear_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = crequests.get(url, headers=headers, impersonate="chrome", timeout=5)
        return int(r.json()['now']['value'])
    except: 
        return 43 # Aktueller Wert als Backup

# --- MAIN DASHBOARD (RE-DESIGNED) ---
st.markdown("## 🌍 Globales Markt-Monitoring")

# Datenabruf (Nutzt deine bestehenden Funktionen)
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
stock_fg = get_cnn_fear_greed() 

# 1. Status-Logik für Farben & Banner
if stock_fg <= 45: 
    fg_col, fg_label = "#e74c3c", "😨 Angst / Panik"
elif stock_fg <= 55: 
    fg_col, fg_label = "#f1c40f", "😐 Neutral"
else: 
    fg_col, fg_label = "#27ae60", "🤑 Gier / Euphorie"

# Markt-Alarm Logik (deine Logik, optisch veredelt)
if dist_ndq < -2 or vix_val > 25:
    m_color, m_text, m_advice = "#e74c3c", "🚨 MARKT-ALARM", "Nasdaq-Schwäche / Hohe Vola. Defensiv agieren!"
elif rsi_ndq > 72 or stock_fg > 80:
    m_color, m_text, m_advice = "#f39c12", "⚠️ ÜBERHITZT", "Korrekturgefahr! Keine neuen Puts mit engem Puffer."
else:
    m_color, m_text, m_advice = "#27ae60", "✅ TRENDSTARK", "Marktumfeld konstruktiv. Puts bei Rücksetzern möglich."

# Der neue interaktive Banner
st.markdown(f"""
    <div style="background: linear-gradient(90deg, {m_color} 0%, #2c3e50 100%); color: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h2 style="margin:0; font-size: 1.6em; letter-spacing: 1px;">{m_text}</h2>
        <p style="margin:5px 0 0 0; font-size: 1.1em; opacity: 0.9;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

# 2. Modernes Kachel-Layout (Grid)
c1, c2, c3 = st.columns(3)

# Hilfsfunktion für Kachel-Styling
def get_card_html(title, value, subtext, color="#3498db"):
    return f"""
    <div style="background-color: #1e2630; border-left: 5px solid {color}; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.3); margin-bottom: 15px;">
        <p style="margin:0; font-size: 0.85em; color: #bdc3c7; text-transform: uppercase; letter-spacing: 1px;">{title}</p>
        <h2 style="margin:5px 0; color: white; font-size: 1.8em;">{value}</h2>
        <p style="margin:0; font-size: 0.9em; color: {color}; font-weight: bold;">{subtext}</p>
    </div>
    """

# Reihe 1
with c1:
    st.markdown(get_card_html("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:+.1f}% vs SMA20", "#3498db"), unsafe_allow_html=True)
with c2:
    st.markdown(get_card_html("CNN Fear & Greed", stock_fg, fg_label, fg_col), unsafe_allow_html=True)
with c3:
    v_col = "#e74c3c" if vix_val > 22 else "#27ae60"
    st.markdown(get_card_html("VIX (Angst)", f"{vix_val:.2f}", "HOCH" if vix_val > 22 else "NORMAL", v_col), unsafe_allow_html=True)

# Reihe 2
c4, c5, c6 = st.columns(3)
with c4:
    st.markdown(get_card_html("Bitcoin", f"{btc_val:,.0f} $", "Digital Gold", "#f39c12"), unsafe_allow_html=True)
with c5:
    st.markdown(get_card_html("Crypto F&G", crypto_fg, "Sentiment", "#9b59b6"), unsafe_allow_html=True)
with c6:
    r_col = "#e74c3c" if rsi_ndq > 70 else "#27ae60"
    st.markdown(get_card_html("Nasdaq RSI", int(rsi_ndq), "HEISS" if rsi_ndq > 70 else "NEUTRAL", r_col), unsafe_allow_html=True)


# --- SEKTION 1: PROFI-SCANNER (TURBO-HYBRID-VERSION) ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro", use_container_width=True):
    # --- 1. BATCH-DOWNLOAD (Phase 1) ---
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("🏁 Phase 1: Batch-Download & Vor-Filterung..."):
        try:
            # SICHERHEITS-CHECK: Verwandelt alles in saubere Strings, entfernt Duplikate
            saubere_liste = list(set([str(t).strip().upper() for t in ticker_liste if t]))
            
            # Download mit der sauberen Liste
            batch_df = yf.download(
                saubere_liste, 
                period="150d", 
                session=session, 
                group_by='ticker', 
                threads=True, 
                progress=False
            )
        except Exception as e:
            st.error(f"Batch-Download fehlgeschlagen: {e}")
            batch_df = pd.DataFrame()

        # Vor-Filterung der Ticker (Preis-Check)
        filtered_tickers = []
        if not batch_df.empty:
            for symbol in saubere_liste:
                try:
                    # Prüfen, ob der Ticker im Download-Ergebnis existiert
                    if symbol not in batch_df.columns.levels[0]: continue
                    
                    s_hist = batch_df[symbol]
                    if s_hist.empty or len(s_hist) < 2: continue
                    
                    last_price = s_hist['Close'].iloc[-1]
                    # Preis-Filter anwenden
                    if min_stock_price <= last_price <= max_stock_price:
                        filtered_tickers.append(symbol)
                except: continue

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
                m_cap = fast.get("market_cap") or fast.get("marketCap") or 0
                
                # PRÜFUNG: Ist es ein ETF? (Nutzt die Liste von oben)
                ist_etf = symbol in etf_tickers
                
                # Filter-Logik: 
                # Abbrechen wenn kein Preis da ist ODER (wenn es kein ETF ist UND das Cap zu klein ist)
                if not price: return None
                if not ist_etf and m_cap < p_min_cap: return None

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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
    with col_header:
        st.subheader("📋 Aktuelle Depot-Übersicht")
    with col_btn:
        if st.button("🔄 Daten aktualisieren", use_container_width=True):
            st.session_state.depot_data_cache = None
            st.rerun()

    # Umwandlung in DataFrame für fortgeschrittenes Styling
    df_depot = pd.DataFrame(st.session_state.depot_data_cache)

    # Hilfsfunktion für das Zeilen-Styling (Heatmap & Signale)
    def style_depot_rows(row):
        # Extrahiere Prozentwert aus dem String "P/L %" (z.B. "+5.2%")
        try:
            perf_val = float(row['P/L %'].replace('%', ''))
        except:
            perf_val = 0
            
        styles = [''] * len(row)
        
        # 1. P/L Heatmap Hintergrund
        if perf_val <= -10:
            styles = ['background-color: rgba(231, 76, 60, 0.2)'] * len(row) # Kräftiges Rot
        elif perf_val < 0:
            styles = ['background-color: rgba(231, 76, 60, 0.05)'] * len(row) # Leichtes Rot
        elif perf_val > 5:
            styles = ['background-color: rgba(46, 204, 113, 0.1)'] * len(row) # Leichtes Grün
            
        # 2. Signal-Hervorhebung (Überschreibt P/L Hintergrund in den Signal-Zellen)
        # Wir können spezifische Spalten-Indizes färben, wenn Signale aktiv sind
        if "JETZT" in str(row['Short Put (Repair)']) or "EXTREM" in str(row['Short Put (Repair)']):
            styles[df_depot.columns.get_loc('Short Put (Repair)')] = 'background-color: #27ae60; color: white; font-weight: bold;'
            
        if "JETZT" in str(row['Covered Call']):
            styles[df_depot.columns.get_loc('Covered Call')] = 'background-color: #2980b9; color: white; font-weight: bold;'
            
        return styles

    # Anzeige als interaktiver, gestylter Dataframe
    st.dataframe(
        df_depot.style.apply(style_depot_rows, axis=1),
        use_container_width=True,
        height=500, # Feste Höhe für bessere Übersicht
        hide_index=True # Index entfernen für sauberen Look
    )

    # --- OPTISCHE LEGENDE ---
    l1, l2, l3, l4 = st.columns(4)
    l1.caption("🔴 **Rot:** Stark im Minus (>10%)")
    l2.caption("🟢 **Grün:** Signal für Short Put")
    l3.caption("🔵 **Blau:** Signal für Covered Call")
    l4.caption("⭐ **Sterne:** Analysten-Rating")

# --- SEKTION 3: PROFI-ANALYSE & TRADING-COCKPIT (FULL VERSION) ---
st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")

# Initialisierung des Session States für stabile Anzeige bei Widget-Interaktion
if 'cockpit_data' not in st.session_state:
    st.session_state.cockpit_data = None

with st.form("cockpit_form"):
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()
    submit_button = st.form_submit_button("🚀 Analyse starten / aktualisieren")

if submit_button and symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input, session=session)
            info = tk.info
            # Nutze die vorhandene Full-Data Funktion aus deinem Skript
            res = get_stock_data_full(symbol_input)
            
            if res[0] is not None:
                # Alles im Session State zwischenspeichern
                st.session_state.cockpit_data = {
                    'symbol': symbol_input,
                    'price': res[0],
                    'dates': res[1],
                    'earn': res[2],
                    'rsi': res[3],
                    'uptrend': res[4],
                    'near_lower': res[5],
                    'atr': res[6],
                    'pivots': res[7],
                    'info': info,
                    'analyst': get_analyst_conviction(info)
                }
    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")

# Anzeige-Logik: Rendert nur, wenn Daten im State vorhanden sind
if st.session_state.cockpit_data:
    d = st.session_state.cockpit_data
    
    # 1. Earnings & Signal-Box
    if d['earn'] and d['earn'] != "---":
        st.info(f"🗓️ Nächste Earnings: {d['earn']}")
    
    s2_d = d['pivots'].get('S2') if d['pivots'] else None
    s2_w = d['pivots'].get('W_S2') if d['pivots'] else None
    
    put_action, sig_col = "⏳ Warten", "white"
    if s2_w and d['price'] <= s2_w * 1.01:
        put_action, sig_col = "🔥 EXTREM (Weekly S2)", "#ff4b4b"
    elif d['rsi'] < 35 or (s2_d and d['price'] <= s2_d * 1.02):
        put_action, sig_col = "🟢 JETZT (S2/RSI)", "#27ae60"

    st.markdown(f'<div style="padding:10px; border-radius:10px; border: 2px solid {sig_col}; text-align:center;"><small>Short Put Signal:</small><br><strong style="font-size:20px; color:{sig_col};">{put_action}</strong></div>', unsafe_allow_html=True)

    # 2. Sterne & Ampel-Logik
    analyst_txt, analyst_col = d['analyst']
    stars = 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1 if "Neutral" in analyst_txt else 0
    if d['uptrend']: stars += 0.5
    
    ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
    if d['rsi'] < 25 or d['rsi'] > 75 or "Warnung" in analyst_txt:
        ampel_color, ampel_text = "#e74c3c", "STOPP / GEFAHR"
    elif stars >= 2.5 and d['uptrend'] and 30 <= d['rsi'] <= 60:
        ampel_color, ampel_text = "#27ae60", "TOP SETUP"

    st.markdown(f'<div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><h2>● {ampel_text}</h2></div>', unsafe_allow_html=True)

    # 3. Metriken & Pivots
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kurs", f"{d['price']:.2f} $")
    c2.metric("RSI", int(d['rsi']))
    c3.metric("Phase", "🛡️ Trend" if d['uptrend'] else "💎 Dip")
    c4.metric("Qualität", "⭐" * int(stars))

    if d['pivots']:
        st.markdown("#### 🛡️ Technische Level")
        pc = st.columns(5)
        pc[0].metric("Weekly S2", f"{d['pivots']['W_S2']:.2f}$")
        pc[1].metric("Daily S2", f"{d['pivots']['S2']:.2f}$")
        pc[2].metric("Pivot P", f"{d['pivots']['P']:.2f}$")
        pc[3].metric("Daily R2", f"{d['pivots']['R2']:.2f}$")
        pc[4].metric("Weekly R2", f"{d['pivots']['W_R2']:.2f}$")

    # 4. OpenClaw KI-Analyse
    ki_status, ki_text, _ = get_openclaw_analysis(d['symbol'])
    st.info(ki_text)

    # 5. SMART OPTIONS-CHAIN MIT EXPECTED MOVE & DELTA
    st.markdown("---")
    st.markdown("### 🎯 Smart Option-Chain & Expected Move")
    opt_mode = st.radio("Strategie:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
    
    heute = datetime.now()
    valid_dates = [dt for dt in d['dates'] if 5 <= (datetime.strptime(dt, '%Y-%m-%d') - heute).days <= 45]
    
    if valid_dates:
        target_date = st.selectbox("📅 Verfallstag wählen", valid_dates)
        days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - heute).days)
        T = days / 365.0
        
        tk_active = yf.Ticker(d['symbol'], session=session)
        opt_chain = tk_active.option_chain(target_date)
        chain = opt_chain.puts if "Put" in opt_mode else opt_chain.calls
        
        # --- EXPECTED MOVE BERECHNUNG ---
        atm_idx = (chain['strike'] - d['price']).abs().idxmin()
        iv_atm = chain.loc[atm_idx, 'impliedVolatility']
        expected_move = d['price'] * iv_atm * np.sqrt(T)
        lower_em = d['price'] - expected_move
        upper_em = d['price'] + expected_move
        
        # EM Anzeige
        em_c1, em_c2 = st.columns(2)
        em_c1.metric("Expected Move (±)", f"{expected_move:.2f} $", f"{iv_atm*100:.1f}% IV")
        em_c2.warning(f"Statistischer Range: **{lower_em:.2f}$ — {upper_em:.2f}$**")

        # Daten Aufbereitung
        df = chain[chain['openInterest'] > 10].copy()
        df['Mid'] = (df['bid'] + df['ask']) / 2
        
        # Delta Berechnung pro Strike
        df['Delta'] = df.apply(lambda x: calculate_bsm_delta(
            d['price'], x['strike'], T, x['impliedVolatility'], option_type='put' if "Put" in opt_mode else 'call'
        ), axis=1)

        if "Put" in opt_mode:
            df = df[df['strike'] < d['price']].sort_values('strike', ascending=False)
            df['Puffer %'] = ((d['price'] - df['strike']) / d['price']) * 100
            df['EM_Safe'] = df['strike'] < lower_em
        else:
            df = df[df['strike'] > d['price']].sort_values('strike', ascending=True)
            df['Puffer %'] = ((df['strike'] - d['price']) / d['price']) * 100
            df['EM_Safe'] = df['strike'] > upper_em
        
        df['Yield p.a. %'] = (df['Mid'] / df['strike']) * (365 / days) * 100

        # Styling
        def style_rows(row):
            styles = [''] * len(row)
            if row['EM_Safe']:
                styles = ['background-color: rgba(16, 185, 129, 0.15)'] * len(row) # Sanftes Grün für EM-Safe
            return styles

        df_show = df[['strike', 'bid', 'ask', 'Mid', 'Delta', 'Puffer %', 'Yield p.a. %', 'EM_Safe']].head(12)
        
        st.dataframe(
            df_show.style.apply(style_rows, axis=1)
            .format({
                'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $', 
                'Mid': '{:.2f} $', 'Delta': '{:.2f}', 'Puffer %': '{:.1f}%', 'Yield p.a. %': '{:.1f}%'
            }), 
            use_container_width=True,
            height=450
        )
        st.caption("🛡️ **Grün hinterlegt:** Strike liegt außerhalb des Expected Move. | **Delta:** Wahrscheinlichkeit für ITM (Ziel: < 0.20)")
