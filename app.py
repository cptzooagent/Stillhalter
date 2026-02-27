import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time

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
    """Berechnet Daily und Weekly Pivot-Punkte (inkl. R2 für CC-Ziele)."""
    try:
        tk = yf.Ticker(symbol)
        
        # 1. Daily Pivots (Vortag)
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s1_d = (2 * p_d) - h_d
        s2_d = p_d - (h_d - l_d)
        r2_d = p_d + (h_d - l_d)  # Widerstand 2 Daily

        # 2. Weekly Pivots (Vorwoche)
        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2: 
            return {
                "P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, 
                "W_S2": s2_d, "W_R2": r2_d
            }
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        s2_w = p_w - (h_w - l_w)
        r2_w = p_w + (h_w - l_w)  # Widerstand 2 Weekly

        return {
            "P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d,
            "W_S2": s2_w, "W_R2": r2_w 
        }
    except:
        return None

# --- HIER EINFÜGEN (ca. Zeile 55) ---
def get_openclaw_analysis(symbol):
    try:
        tk = yf.Ticker(symbol)
        all_news = tk.news
        
        if not all_news or len(all_news) == 0:
            return "Neutral", "🤖 OpenClaw: Yahoo liefert aktuell keine Daten.", 0.5
        
        huge_blob = str(all_news).lower()
        display_text = ""

        # Wir suchen jetzt nach JEDEM String, der mindestens 3 Leerzeichen hat 
        # (ein sicheres Zeichen für einen Satz, keine ID)
        for n in all_news:
            for val in n.values():
                if isinstance(val, str) and val.count(" ") > 3:
                    display_text = val
                    break
            if display_text: break

        if not display_text:
            # Fallback: Falls gar kein Satz gefunden wird, nimm das Feld 'title' direkt
            display_text = all_news[0].get('title', 'Marktstimmung aktiv (Text folgt)')

        # Sentiment-Logik
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

    except Exception:
        return "N/A", "🤖 OpenClaw: System-Reset...", 0.5
        
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
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

# --- 2. DATEN-FUNKTIONEN (REPARATUR BILD 5) ---
@st.cache_data(ttl=3600)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
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
        
        # WICHTIG: Pivots berechnen
        pivots = calculate_pivots(symbol)
        
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        # Rückgabe von 8 Werten im Erfolgsfall
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr, pivots
    except:
        # Rückgabe von 8 Werten im Fehlerfall (ZUSÄTZLICHES 'None' am Ende!)
        return None, [], "", 50, True, False, 0, None
        
# --- UI: SIDEBAR (KOMPLETT-REPARATUR) ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    
    # Basis-Filter
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 15, key="puffer_sid")
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12, key="yield_sid")
    
    # Der Aktienpreis-Regler
    min_stock_price, max_stock_price = st.slider(
        "Aktienpreis-Spanne ($)", 
        0, 1000, (60, 500), 
        key="price_sid"
    )

    st.markdown("---")
    st.subheader("Qualitäts-Filter")
    
    # Marktkapitalisierung & Trend
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 1, 1000, 20, key="mkt_cap_sid")
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    
    # WICHTIG: Die Checkbox für den Simulationsmodus
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")
    
    st.markdown("---")
    st.info("💡 Profi-Tipp: Für den S&P 500 Scan ab 16:00 Uhr 'Simulations-Modus' deaktivieren.")
    
# --- HILFSFUNKTIONEN FÜR DAS DASHBOARD ---

def get_market_data():
    """Holt globale Marktdaten mit korrektem Nasdaq 100 Index (^NDX)."""
    try:
        ndq = yf.Ticker("^NDX")
        vix = yf.Ticker("^VIX")
        btc = yf.Ticker("BTC-USD")
        
        h_ndq = ndq.history(period="1mo")
        h_vix = vix.history(period="1d")
        h_btc = btc.history(period="1d")
        
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
    except:
        return 0, 50, 0, 20, 0

def get_crypto_fg():
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/")
        return int(r.json()['data'][0]['value'])
    except:
        return 50

# --- HAUPT-BLOCK: GLOBAL MONITORING ---

st.markdown("## 🌍 Globales Markt-Monitoring")

# Daten abrufen
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
# Hinweis: fg_val (Stock) müsstest du aus deiner vorhandenen Funktion nehmen
stock_fg = 50 # Platzhalter, falls deine Funktion anders heißt

# Master-Banner Logik
if dist_ndq < -2 or vix_val > 25:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Nasdaq-Schwäche / Hohe Volatilität"
    m_advice = "Defensiv agieren. Fokus auf Call-Verkäufe zur Depot-Absicherung."
elif rsi_ndq > 72 or stock_fg > 80:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Korrekturgefahr (Gier/RSI hoch)"
    m_advice = "Keine neuen Puts mit engem Puffer. Gewinne sichern."
else:
    m_color, m_text = "#27ae60", "✅ TRENDSTARK: Marktumfeld ist konstruktiv"
    m_advice = "Puts auf starke Aktien bei Rücksetzern möglich."

st.markdown(f"""
    <div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; font-size: 1.4em;">{m_text}</h3>
        <p style="margin:0; opacity: 0.9;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

# Das 2x3 Raster
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1:
    st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2:
    st.metric("Bitcoin", f"{btc_val:,.0f} $")
with r1c3:
    st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")

with r2c1:
    st.metric("Fear & Greed (Stock)", f"{stock_fg}")
with r2c2:
    st.metric("Fear & Greed (Crypto)", f"{crypto_fg}")
with r2c3:
    st.metric("Nasdaq RSI (14)", f"{int(rsi_ndq)}", delta="HEISS" if rsi_ndq > 70 else None, delta_color="inverse")

st.markdown("---")


# --- NEUE ANALYSTEN-LOGIK (VOR DEM SCAN DEFINIEREN) ---
def get_analyst_conviction(info):
    try:
        current = info.get('current_price', info.get('currentPrice', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        
        # 1. LILA: 🚀 HYPER-GROWTH (APP, CRDO, NVDA, ALAB)
        if rev_growth > 40:
            return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        # 2. GRÜN: ✅ STARK (AVGO, CRWD)
        elif upside > 15 and rev_growth > 5:
            return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        # 3. BLAU: 💎 QUALITY-DIP (NET, MRVL)
        elif upside > 25:
            return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        # 4. ORANGE: ⚠️ WARNUNG (GTM, stagnierende Werte)
        elif upside < 0 or rev_growth < -2:
            return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except:
        return "🔍 Check nötig", "#7f8c8d"



# --- SEKTION 1: PROFI-SCANNER (ULTRA-SPEED FAST_INFO EDITION) ---

# 1. Speicher initialisieren
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten (High Speed)", key="kombi_scan_pro"):
    puffer_limit = otm_puffer_slider / 100 
    mkt_cap_limit = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Markt-Scanner läuft (Fast-Info Modus aktiv)..."):
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []

    # --- DIE OPTIMIERTE UNTER-FUNKTION ---
    def check_single_stock(symbol):
        try:
            tk = yf.Ticker(symbol)
            
            # SCHRITT A: FAST_INFO (Extrem schnell für Basis-Filter)
            # Verhindert unnötige .info Abfragen für unpassende Aktien
            f_info = tk.fast_info
            curr_price = f_info.get('last_price') or f_info.get('lastPrice')
            m_cap = f_info.get('market_cap') or f_info.get('marketCap')

            if not curr_price or m_cap < mkt_cap_limit: return None
            if not (min_stock_price <= curr_price <= max_stock_price): return None
            
            # SCHRITT B: VOLLSTÄNDIGE DATEN (Nur wenn Filter A bestanden)
            res = get_stock_data_full(symbol)
            if res[0] is None: return None
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
            
            if only_uptrend and not uptrend: return None

            # Earnings-Check (Laufzeit-Schutz)
            max_days_allowed = 24
            if earn and "." in earn:
                try:
                    tag, monat = earn.split(".")[:2]
                    er_datum = datetime(heute.year, int(monat), int(tag))
                    if er_datum < heute: er_datum = datetime(heute.year + 1, int(monat), int(tag))
                    max_days_allowed = min(24, (er_datum - heute).days - 2)
                except: pass

            # Options-Check
            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= max_days_allowed]
            if not valid_dates: return None
            
            target_date = valid_dates[0]
            chain = tk.option_chain(target_date).puts
            target_strike = price * (1 - puffer_limit)
            opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not opts.empty:
                o = opts.iloc[0]
                bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                days = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (bid_val / o['strike']) * (365 / max(1, days)) * 100
                
                if y_pa >= min_yield_pa:
                    # SCHRITT C: FUNDAMENTAL-DATEN (Nur für finale Kandidaten)
                    info = tk.info
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    
                    stars = 0
                    if "HYPER" in analyst_txt: stars = 3
                    elif "Stark" in analyst_txt: stars = 2
                    elif "Neutral" in analyst_txt: stars = 1
                    
                    if rsi < 30: stars -= 1 
                    if rsi > 75: stars -= 0.5 
                    if uptrend and stars > 0: stars += 0.5 
                    stars = max(0, float(stars))

                    return {
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                        'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                        'bid': bid_val, 'rsi': rsi, 'earn': earn if earn else "n.a.", 
                        'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                        'stars_val': stars, 'stars_str': "⭐" * int(stars) if stars >= 1 else "⚠️",
                        'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                        'mkt_cap': m_cap / 1_000_000_000
                    }
        except: return None
        return None

    # Parallelisierung mit moderater Worker-Anzahl für Yahoo Stability
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            current_ticker = futures[future] 
            res_data = future.result()
            if res_data:
                all_results.append(res_data)
            
            progress_bar.progress((i + 1) / len(ticker_liste))
            if i % 5 == 0:
                status_text.text(f"Scanning {i}/{len(ticker_liste)}: {current_ticker}...")

    status_text.empty()
    progress_bar.empty()

    if all_results:
        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)
    else:
        st.session_state.profi_scan_results = []
        st.warning("Keine Treffer unter den aktuellen Kriterien gefunden.")

# --- RESULTATE ANZEIGEN (Außerhalb des Buttons, damit sie stehen bleiben) ---
if st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
    
    # Optional: Button zum Löschen der Ergebnisse
    if st.button("Ergebnisse löschen"):
        st.session_state.profi_scan_results = []
        st.rerun()

    cols = st.columns(4)
    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            s_color = "#27ae60" if "🛡️" in res['status'] else "#2980b9"
            border_color = res['analyst_col'] if res['stars_val'] >= 2 else "#e0e0e0"
            rsi_col = "#e74c3c" if res['rsi'] > 70 or res['rsi'] < 30 else "#7f8c8d"
            
            with st.container(border=True):
                st.markdown(f"**{res['symbol']}** {res['stars_str']} <span style='float:right; font-size:0.75em; color:{s_color}; font-weight:bold;'>{res['status']}</span>", unsafe_allow_html=True)
                st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                
                st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; border: 2px solid {border_color}; margin-bottom: 8px; font-size: 0.85em;">
                        🎯 Strike: <b>{res['strike']:.1f}$</b> | 💰 Bid: <b>{res['bid']:.2f}$</b><br>
                        🛡️ Puffer: <b>{res['puffer']:.1f}%</b> | ⏳ Tage: <b>{res['tage']}</b>
                    </div>
                    <div style="font-size: 0.8em; color: #7f8c8d; margin-bottom: 5px;">
                        📅 ER: <b>{res['earn']}</b> | RSI: <b style="color:{rsi_col};">{int(res['rsi'])}</b>
                    </div>
                    <div style="font-size: 0.85em; border-left: 4px solid {res['analyst_col']}; padding: 4px 8px; font-weight: bold; color: {res['analyst_col']}; background: {res['analyst_col']}10; border-radius: 0 4px 4px 0;">
                        {res['analyst_txt']}
                    </div>
                """, unsafe_allow_html=True)
                    
# --- SEKTION 2: DEPOT-MANAGER (STABILISIERTE VERSION) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

# DER BUTTON: Nur wenn dieser gedrückt wird, läuft der folgende Block ab
if st.button("🚀 Depot-Daten jetzt laden/aktualisieren"):

my_assets = {
    "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
    "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
    "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
    "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
}

with st.expander("📂 Mein Depot & Strategie-Signale", expanded=True):
    depot_list = []
    for symbol, data in my_assets.items():
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            
            # 1. Daten entpacken
            price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
            qty, entry = data[0], data[1]
            perf_pct = ((price - entry) / entry) * 100

            # 2. KI-Stimmung (OpenClaw) - NUR EINMAL AUFRUFEN
            ki_status, ki_text, _ = get_openclaw_analysis(symbol)
            ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"

            # 3. Sterne-Rating (Optimiert mit Try-Except für Speed)
            try:
                # Wir holen info nur einmal
                info_temp = yf.Ticker(symbol).info
                analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                # Kompakte Sterne-Zuweisung
                stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                star_display = "⭐" * stars_count
            except:
                star_display = "⭐" # Fallback auf 1 Stern bei Fehler

            # 4. Pivot-Werte sicher extrahieren (ACHTUNG: Unterstrich nutzen!)
            r2_d = pivots.get('R2') if pivots else None
            r2_w = pivots.get('W_R2') if pivots else None
            s2_d = pivots.get('S2') if pivots else None
            s2_w = pivots.get('W_S2') if pivots else None
            
            # --- Hier folgt deine bestehende Put/Call Action Logik ---
            # ... (put_action = ..., call_action = ...)
            
            # Reparatur-Logik (Put)
            put_action = "⏳ Warten"
            if rsi < 35 or (s2_d and price <= s2_d * 1.02):
                put_action = "🟢 JETZT (S2/RSI)"
            if s2_w and price <= s2_w * 1.01:
                put_action = "🔥 EXTREM (Weekly S2)"
            
            # Covered Call Logik (NUR GRÜN WENN R2 EXISTIERT UND > 0 IST)
            call_action = "⏳ Warten"
            if rsi > 55:
                if r2_d and price >= r2_d * 0.98:
                    call_action = "🟢 JETZT (R2/RSI)"
                else:
                    call_action = "🟡 RSI HOCH (Warte auf R2)"

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
                "S2 Weekly": f"{s2_w:.2f} $" if s2_w else "---", # JETZT DABEI
                "R2 Daily": f"{r2_d:.2f} $" if r2_d else "---",
                "R2 Weekly": f"{r2_w:.2f} $" if r2_w else "---"  # JETZT DABEI
            })
        except: continue
    
    if depot_list:
        st.table(pd.DataFrame(depot_list))
        
st.info("💡 **Strategie:** Wenn 'Short Put' auf 🔥 steht, ist die Aktie am wöchentlichen Tiefstand – technisch das sicherste Level zum Verbilligen.")
                    
# --- SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (INKL. PANIK-SCHUTZ) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)

            # Ändere diese Zeile in Sektion 3:
            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res  # pivots_res hinzugefügt
                analyst_txt, analyst_col = get_analyst_conviction(info)

                # --- NEU: Earnings-Anzeige im Scanner (vor der Ampel) ---
                if earn and earn != "---":
                    # Optischer Hinweis, falls Earnings in Kürze anstehen (Beispiel Feb/März 2026)
                    if "Feb" in earn or "Mar" in earn:
                        st.error(f"⚠️ **Earnings-Warnung:** Nächste Zahlen am {earn}. Vorsicht bei neuen Trades!")
                    else:
                        st.info(f"🗓️ Nächste Earnings: {earn}")
                else:
                    st.write("🗓️ Keine Earnings-Daten verfügbar")

                # --- 2. NEU: STRATEGIE-SIGNAL (Die Logik aus dem Depot-Manager) ---
                # Wir extrahieren die S2-Werte aus pivots_res für die Berechnung
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
        
                put_action_scanner = "⏳ Warten (Kein Signal)"
                signal_color = "white"

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b" # Rot für Alarm/Chance
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60" # Grün für Einstieg

                # Anzeige des Signals als hervorgehobene Metrik
                st.markdown(f"""
                    <div style="padding:10px; border-radius:10px; border: 2px solid {signal_color}; text-align:center;">
                        <small>Aktuelles Short Put Signal:</small><br>
                        <strong style="font-size:20px; color:{signal_color};">{put_action_scanner}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Sterne-Logik (Basis für Qualität)
                stars = 0
                if "HYPER" in analyst_txt: stars = 3
                elif "Stark" in analyst_txt: stars = 2
                elif "Neutral" in analyst_txt: stars = 1
                if uptrend and stars > 0: stars += 0.5
                
                # --- VERSCHÄRFTE AMPEL-LOGIK (PANIK-SCHUTZ) ---
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    # Panik-Schutz greift zuerst
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif stars >= 2.5 and uptrend and 30 <= rsi <= 60:
                    # Ideales Setup
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ANALYSTEN-WARNUNG"
                else:
                    ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"

                # 1. HEADER: Ampel
                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="margin:0; font-size: 1.8em; letter-spacing: 1px;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # 2. METRIKEN-BOARD
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Kurs", f"{price:.2f} $")
                with col2:
                    st.metric("RSI (14)", f"{int(rsi)}", delta="PANIK" if rsi < 25 else None, delta_color="inverse")
                with col3:
                    status_icon = "🛡️" if uptrend else "💎"
                    st.metric("Phase", f"{status_icon} {'Trend' if uptrend else 'Dip'}")
                with col4:
                    st.metric("Qualität", "⭐" * int(stars))

                # --- VOLLSTÄNDIGE PIVOT ANALYSE ANZEIGE ---
                st.markdown("---")
                pivots = calculate_pivots(symbol_input)
                if pivots:
                    st.markdown("#### 🛡️ Technische Absicherung & Ziele (Pivots)")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    
                    pc1.metric("Weekly S2 (Boden)", f"{pivots['W_S2']:.2f} $")
                    pc2.metric("Daily S2", f"{pivots['S2']:.2f} $")
                    pc3.metric("Pivot (P)", f"{pivots['P']:.2f} $")
                    pc4.metric("Daily R2 (Ziel)", f"{pivots['R2']:.2f} $")
                    pc5.metric("Weekly R2 (Top)", f"{pivots['W_R2']:.2f} $")
                    
                    st.caption(f"💡 **CC-Tipp:** Ein Covered Call am R2 Weekly ({pivots['W_R2']:.2f} $) bietet die höchste statistische Sicherheit gegen Ausstoppen.")

                # 3. ANALYSTEN BOX
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                        <hr style="margin: 10px 0;">
                        <span style="color: #555;">📅 Nächste Earnings: <b>{earn if earn else 'n.a.'}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # --- 4. OPTIONEN TABELLE & UMSCHALTER (AKTUALISIERTE VERSION) ---
                st.markdown("---")
                st.markdown("### 🎯 Option-Chain Auswahl")
                
                # Der neue Umschalter
                option_mode = st.radio("Strategie wählen:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                heute = datetime.now()
                # Zeitfenster: 5 bis 35 Tage
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days

                    # NEU: OpenClaw KI-Box VOR der Tabelle
                    ki_status, ki_text, ki_score = get_openclaw_analysis(symbol_input)
                    st.info(ki_text) # Zeigt die KI-News direkt an

                    opt_chain = tk.option_chain(target_date)
                    chain = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    df_disp = chain.copy()

                    # NEU: Filter für Liquidität (Open Interest)
                    df_disp = df_disp[df_disp['openInterest'] > 50]
                    
                    # Holen der Daten je nach Modus
                    if "Put" in option_mode:
                        chain = tk.option_chain(target_date).puts
                        # Filter: Nur Strikes unter aktuellem Preis
                        df_disp = chain[chain['strike'] < price].copy()
                        df_disp['Puffer %'] = ((price - df_disp['strike']) / price) * 100
                        sort_order = False # Höchster Strike zuerst
                    else:
                        chain = tk.option_chain(target_date).calls
                        # Filter: Nur Strikes über aktuellem Preis
                        df_disp = chain[chain['strike'] > price].copy()
                        df_disp['Puffer %'] = ((df_disp['strike'] - price) / price) * 100
                        sort_order = True # Niedrigster Strike (über Preis) zuerst

                    # Berechnungen
                    df_disp['strike'] = df_disp['strike'].astype(float)
                    df_disp['Yield p.a. %'] = (df_disp['bid'] / df_disp['strike']) * (365 / max(1, days_to_expiry)) * 100
                    
                    # Sortierung für bessere Übersicht
                    df_disp = df_disp.sort_values('strike', ascending=sort_order)

                    # Styling Funktion
                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 5 <= p < 10: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    # Tabelle anzeigen (Top 15 Strikes)
                    styled_df = df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(15).style.apply(style_rows, axis=1).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                        'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True, height=450)
                    
                    # Dynamische Legende
                    if "Put" in option_mode:
                        st.caption("🟢 >10% Puffer (Sicherer) | 🟡 5-10% | 🔴 <5% (Aggressiv)")
                    else:
                        st.caption("🟢 >10% Abstand (Konservativer Call) | 🟡 5-10% | 🔴 <5% (Hohes Ausbuchungs-Risiko)")

    except Exception as e:
        st.error(f"Fehler bei {symbol_input}: {e}")
