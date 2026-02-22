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
# --- DIESE FUNKTION ÜBER get_stock_data_full PLATZIEREN ---
def get_finviz_sentiment(symbol):
    try:
        # Kurzer Hack für die Simulation, damit Icons kommen:
        # Wenn du echtes Scraping willst, nimm den Block von vorhin
        import random
        return random.choice(["🟢", "🟡", "🟢"]), 0.2 
    except:
        return "⚪", 0.0

# --- IN check_single_stock ANPASSEN ---
# Suche die Stelle, an der s_val berechnet wird und ersetze sie hiermit:
    
    # 1. Sentiment holen
    sent_icon, sent_score = get_finviz_sentiment(symbol)
    
    # 2. Sterne-Logik verfeinern
    analyst_txt, analyst_col = get_analyst_conviction(info)
    s_val = 0.0
    
    # Punkte für Analysten
    if "HYPER" in analyst_txt: s_val += 2.5
    elif "Stark" in analyst_txt: s_val += 1.5
    elif "Quality" in analyst_txt: s_val += 1.0
    
    # Punkte für Technik
    if rsi < 40: s_val += 0.5  # Überverkauft ist gut für Puts
    if uptrend: s_val += 0.5   # Trendfolge gibt Sicherheit
    
    # Bonus für positives Sentiment
    if sent_icon == "🟢": s_val += 0.5
        
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



# --- SEKTION 1: PROFI-SCANNER (MIT EXPECTED MOVE & SESSION-FIX) ---

# 1. Speicher initialisieren
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    # Filter-Werte aus den Slidern ziehen
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Markt-Scanner analysiert Ticker..."):
        # Ticker-Liste bestimmen
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste = get_combined_watchlist()
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        all_results = []

        # --- DIE OPTIMIERTE UNTER-FUNKTION (MIT EM-LOGIK) ---
        def check_single_stock(symbol):
            try:
                # API-Abfrage mit Session-Sicherung (Tarnkappe)
                time.sleep(random.uniform(0.2, 0.4)) 
                tk = yf.Ticker(symbol, session=custom_session)
                info = tk.info
                if not info or 'currentPrice' not in info: return None
        
                m_cap = info.get('marketCap', 0)
                price = info.get('currentPrice', 0)
        
                # Filter-Logik
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): 
                    return None

                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: return None
                _, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
        
                if only_uptrend and not uptrend: return None

                # Laufzeit-Filter (10 bis 30 Tage)
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 30]
                if not valid_dates: return None
        
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
        
                if opts.empty: return None

                o = opts.iloc[0]
                bid, ask = o.get('bid', 0), o.get('ask', 0)
                fair_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else o.get('lastPrice', 0)
        
                # --- MATHEMATIK: EM & DELTA ---
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                iv = o.get('impliedVolatility', 0.4)
                
                # Expected Move (EM) in %: IV * SQRT(Tage/365)
                em_pct = (iv * np.sqrt(max(0.01, days_to_exp) / 365)) * 100
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv, option_type='put')
        
                sent_icon, _ = get_finviz_sentiment(symbol)
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
        
                if y_pa >= p_min_yield:
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    s_val = 0.0
                    if "HYPER" in analyst_txt: s_val = 3.0
                    elif "Stark" in analyst_txt: s_val = 2.0
                    if rsi < 35: s_val += 0.5
                    if uptrend: s_val += 0.5

                    return {
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                        'puffer': ((price - o['strike']) / price) * 100, 'bid': fair_price,
                        'rsi': rsi, 'earn': earn if earn else "---", 'tage': days_to_exp, 
                        'status': "🛡️ Trend" if uptrend else "💎 Dip", 'delta': delta_val,
                        'sent_icon': sent_icon, 'stars_val': s_val, 
                        'stars_str': "⭐" * int(s_val) if s_val >= 1 else "⚠️",
                        'analyst_label': analyst_txt, 'analyst_color': analyst_col,
                        'mkt_cap': m_cap / 1e9, 'em_pct': em_pct
                    }
            except: return None

        # Multithreading Start
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data:
                    all_results.append(res_data)
                
                progress_bar.progress((i + 1) / len(ticker_liste))
                if i % 5 == 0:
                    status_text.text(f"Checke {i}/{len(ticker_liste)} Ticker...")

        status_text.empty()
        progress_bar.empty()

        if all_results:
            st.session_state.profi_scan_results = sorted(
                all_results, 
                key=lambda x: (float(x.get('stars_val', 0) or 0), float(x.get('y_pa', 0) or 0)), 
                reverse=True
            )
            st.success(f"Scan abgeschlossen: {len(all_results)} Treffer gefunden!")
        else:
            st.session_state.profi_scan_results = []
            st.warning("Keine Treffer gefunden. API-Blockade oder Puffer zu hoch.")

# =========================================================
# --- SEKTION: RESULTATE ANZEIGEN (MIT EM-VISUALISIERUNG) ---
# =========================================================
if 'profi_scan_results' in st.session_state and st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(all_results)} Treffer)")
    
    cols = st.columns(4)
    heute_dt = datetime.now()

    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            # Daten-Vorbereitung
            earn_str = res.get('earn', "---")
            status_txt = res.get('status', "Trend")
            sent_icon = res.get('sent_icon', "🟢")
            stars = res.get('stars_str', "⭐")
            s_color = "#10b981" if "Trend" in status_txt else "#3b82f6"
            a_label = res.get('analyst_label', "Keine Analyse")
            a_color = res.get('analyst_color', "#8b5cf6")
            mkt_cap = res.get('mkt_cap', 0)
            
            # RSI & Delta Styling
            rsi_val = int(res.get('rsi', 50))
            rsi_style = "color: #ef4444;" if rsi_val >= 70 else "color: #10b981;" if rsi_val <= 35 else "color: #4b5563;"
            delta_val = abs(res.get('delta', 0))
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            
            # --- EXPECTED MOVE LOGIK ---
            em_val = res.get('em_pct', 0)
            puffer_val = res.get('puffer', 0)
            em_color = "#10b981" if puffer_val > em_val else "#f59e0b" if puffer_val > (em_val * 0.8) else "#ef4444"
            progress = min(100, (em_val / puffer_val * 100)) if puffer_val > 0 else 100

            # Earnings Risiko Check
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    earn_date = datetime.strptime(f"{earn_str}2026", "%d.%m.%Y")
                    if 0 <= (earn_date - heute_dt).days <= res.get('tage', 14):
                        is_earning_risk = True
                except: pass

            card_border = "4px solid #ef4444" if is_earning_risk else "1px solid #e5e7eb"
            
            # HTML-LAYOUT
            html_code = f"""
<div style="background: white; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.2em; font-weight: 800; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.8em;">{stars}</span></span>
<span style="font-size: 0.75em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 2px 8px; border-radius: 6px;">{sent_icon} {status_txt}</span>
</div>
<div style="margin: 10px 0;">
<div style="font-size: 0.7em; color: #6b7280; font-weight: 600; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 1.9em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;">
<div style="font-size: 0.6em; color: #6b7280;">Strike</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['strike']:.1f}$</div>
</div>
<div style="border-left: 3px solid {delta_col}; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Delta</div>
<div style="font-size: 0.9em; font-weight: 700; color: {delta_col};">{delta_val:.2f}</div>
</div>
</div>

<div style="background: #f9fafb; border-radius: 10px; padding: 10px; margin-bottom: 12px; border: 1px solid #f3f4f6;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
        <span style="font-size: 0.65em; color: #6b7280; font-weight: 800;">EXPECTED MOVE (1σ)</span>
        <span style="font-size: 0.8em; font-weight: 800; color: {em_color};">±{em_val:.1f}%</span>
    </div>
    <div style="height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden;">
        <div style="width: {progress}%; height: 100%; background: {em_color};"></div>
    </div>
    <div style="font-size: 0.55em; color: #9ca3af; margin-top: 4px; text-align: right;">
        Puffer-Schutz: {(puffer_val/max(0.1, em_val)):.1f}x
    </div>
</div>

<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<span style="{rsi_style}">RSI: {rsi_val}</span>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">
{'⚠️' if is_earning_risk else '🗓️'} {earn_str}
</span>
</div>
<div style="background: {a_color}10; color: {a_color}; padding: 8px; border-radius: 8px; border-left: 4px solid {a_color}; font-size: 0.7em; font-weight: bold; text-align: center;">
{a_label}
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)
else:
    st.info("Scanner bereit. Bitte auf '🚀 Profi-Scan starten' klicken.")
                    
# --- SEKTION 2: DEPOT-MANAGER (MIT PIVOT-PUNKTEN) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell pausiert, um den Start zu beschleunigen.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
        with st.spinner("Berechne Pivot-Punkte und Signale..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
                "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    time.sleep(0.6) 
                    res = get_stock_data_full(symbol)
                    if res is None or res[0] is None: continue
                    
                    price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # Sterne & KI
                    ki_status, _, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
                    
                    try:
                        info_temp = yf.Ticker(symbol).info
                        analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                        stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                        star_display = "⭐" * stars_count
                    except:
                        star_display = "⭐"

                    # PIVOT DATEN EXTRAHIEREN
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    r2_w = pivots.get('W_R2') if pivots else None
                    
                    # AKTIONEN LOGIK
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"
                    
                    call_action = "⏳ Warten"
                    if rsi > 55 and r2_d and price >= r2_d * 0.98: call_action = "🟢 JETZT (R2/RSI)"

                    # DATENSATZ ERSTELLEN
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
    col_header, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Daten aktualisieren"):
            st.session_state.depot_data_cache = None
            st.rerun()

    # Anzeige der Tabelle mit allen Pivot-Spalten
    st.table(pd.DataFrame(st.session_state.depot_data_cache))
                    
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
                    
                    st.dataframe(
                        styled_df, 
                        use_container_width=True, 
                        height=400
                    )
                    
                    st.caption("🟢 >10% Puffer | 🟡 5-10% Puffer | 🔴 <5% Puffer (Risiko)")

    except Exception as e:
        st.error(f"Fehler bei der Detail-Analyse: {e}")
        st.info("Hinweis: Manche Ticker-Symbole liefern am Wochenende oder bei geringer Liquidität keine Optionsdaten.")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Datenquelle: Yahoo Finance | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live-Scan'}")

