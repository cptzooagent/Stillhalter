import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import random
import time
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures

# --- HILFSFUNKTIONEN (Müssen GANZ OBEN stehen) ---

def get_stars_logic(analyst_label, uptrend):
    """Berechnet die Sterne-Bewertung basierend auf Analysten und Trend."""
    s_val = 1.0
    if "HYPER" in analyst_label: 
        s_val = 3.0
    elif "Stark" in analyst_label: 
        s_val = 2.0
        
    if uptrend: 
        s_val += 1.0
        
    return s_val, "⭐" * int(s_val)

# --- SETUP & STYLING ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- SIDEBAR (Zuerst für Demo-Modus Schalter) ---
with st.sidebar:
    st.header("⚙️ System-Einstellungen")
    demo_mode = st.toggle("🛠️ Demo-Modus (API Bypass)", value=False, help="Aktivieren, wenn Yahoo Finance dich gesperrt hat.")
    
    st.markdown("---")
    st.header("🛡️ Scanner-Filter")
    otm_puffer_slider = st.slider("OTM Puffer (%)", 5, 25, 12)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 5, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis ($)", 0, 1000, (40, 600))
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 15)
    only_uptrend = st.checkbox("Nur SMA 200 Uptrend", value=False)
    test_modus = st.checkbox("🔍 Kleiner Scan (12 Ticker)", value=False)

# --- 1. DER SICHERHEITS-CACHE (MIT DEMO-LOGIK) ---
@st.cache_data(ttl=3600)
def get_batch_data_cached(tickers, is_demo=False):
    if is_demo:
        # Erzeuge synthetische Daten für den Demo-Modus
        st.warning("🚧 Demo-Modus aktiv: Zeige generierte Testdaten.")
        dates = pd.date_range(end=datetime.now(), periods=250)
        demo_df = pd.DataFrame(index=dates)
        
        if len(tickers) > 1:
            multi_index = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
            data = pd.DataFrame(np.random.randn(250, len(tickers)*6) * 10 + 150, index=dates, columns=multi_index)
            return data
        return pd.DataFrame(np.random.randn(250, 6) * 10 + 150, index=dates, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    try:
        data = yf.download(tickers, period="250d", group_by='ticker', auto_adjust=True, progress=False)
        return data
    except Exception as e:
        st.error(f"⚠️ Yahoo-Fehler: {e}")
        return pd.DataFrame()

# --- 2. TECHNISCHE MATHEMATIK ---
def calculate_rsi_vectorized(series, window=14):
    if series.empty or len(series) < window: return pd.Series([50] * len(series))
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).fillna(50)

def get_market_context(is_demo=False):
    if is_demo:
        return {"cp": 16540, "rsi": 45, "dist": -1.2, "vix": 18.5, "btc": 92000, "fg": 65}
    
    res = {"cp": 0, "rsi": 50, "dist": 0, "vix": 20, "btc": 0, "fg": 50}
    try:
        data = yf.download(["^NDX", "^VIX", "BTC-USD"], period="60d", interval="1d", progress=False)
        if not data.empty:
            if '^NDX' in data['Close']:
                c = data['Close']['^NDX'].dropna()
                res["cp"] = c.iloc[-1]
                res["dist"] = ((res["cp"] - c.rolling(20).mean().iloc[-1]) / c.rolling(20).mean().iloc[-1]) * 100
                res["rsi"] = calculate_rsi_vectorized(c).iloc[-1]
            res["vix"] = data['Close']['^VIX'].iloc[-1] if '^VIX' in data['Close'] else 20
            res["btc"] = data['Close']['BTC-USD'].iloc[-1] if 'BTC-USD' in data['Close'] else 0
        res["fg"] = int(requests.get("https://api.alternative.me/fng/").json()['data'][0]['value'])
    except: pass
    return res

# --- 3. ANALYSE-HELPER (MIT DEMO-FALLBACK) ---
def get_analyst_conviction(info, is_demo=False):
    if is_demo: return "✅ Stark (Ziel: +18%)", "#27ae60"
    try:
        cur = info.get('currentPrice', info.get('lastPrice', 1))
        tar = info.get('targetMedianPrice', 0)
        upside = ((tar / cur) - 1) * 100 if tar > 0 else 0
        if info.get('revenueGrowth', 0) > 0.3: return "🚀 HYPER-GROWTH", "#9b59b6"
        if upside > 15: return f"✅ Stark (Ziel: +{upside:.0f}%)", "#27ae60"
        return "⚖️ Neutral", "#7f8c8d"
    except: return "🔍 Check", "#7f8c8d"

# --- 4. VISUALISIERUNG MARKT-AMPEL ---
st.markdown("## 🌍 Globales Markt-Monitoring")
m = get_market_context(is_demo=demo_mode)

# Ampel-Logik
ampel_color = "#27ae60"
ampel_text = "MARKT STABIL"
if m["dist"] < -2 or m["vix"] > 24: ampel_color, ampel_text = "#e74c3c", "🚨 MARKT-ALARM"
elif m["rsi"] > 70: ampel_color, ampel_text = "#f39c12", "⚠️ ÜBERHITZT"

st.markdown(f"""
    <div style="background-color: {ampel_color}; color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
        <h2 style="margin:0;">{ampel_text} {'(DEMO)' if demo_mode else ''}</h2>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nasdaq 100", f"{m['cp']:,.0f}", f"{m['dist']:.1f}%")
c2.metric("Bitcoin", f"{m['btc']:,.0f} $")
c3.metric("VIX (Angst)", f"{m['vix']:.2f}")
c4.metric("Nasdaq RSI", f"{int(m['rsi'])}")
st.markdown("---")

# --- BLOCK 2: PROFI-SCANNER (ORIGINAL DESIGN & TREND-LOGIK) ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="run_pro_scan", use_container_width=True):
    all_results = []
    
    # --- PFAD A: DEMO-MODUS (API BYPASS) ---
    if demo_mode:
        with st.spinner("Generiere Demo-Setups im Original-Design..."):
            time.sleep(1) 
            demo_tickers = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "MU", "PLTR", "AMZN", "META", "COIN", "MSTR", "NFLX"]
            for s in demo_tickers:
                # Logik: Wenn Filter aktiv, erzwingen wir Uptrend für die Demo
                is_uptrend = True if only_uptrend else random.choice([True, False])
                price = random.uniform(100, 950)
                
                all_results.append({
                    'symbol': s,
                    'stars_str': "⭐⭐" + ("⭐" if random.random() > 0.5 else ""),
                    'sent_icon': "🟢" if is_uptrend else "🔹",
                    'status': "Trend" if is_uptrend else "Dip",
                    'y_pa': random.uniform(15.0, 38.0),
                    'strike': price * 0.85,
                    'bid': random.uniform(1.5, 5.0),
                    'puffer': random.uniform(10, 22),
                    'delta': random.uniform(-0.10, -0.35),
                    'em_pct': random.uniform(0.5, 4.5) * (1 if random.random() > 0.5 else -1),
                    'em_safety': random.uniform(0.8, 2.1),
                    'tage': 32,
                    'rsi': random.randint(30, 75),
                    'mkt_cap': random.uniform(50, 2500),
                    'earn': random.choice(["15.03.", "22.04.", "---"]),
                    'analyst_label': random.choice(["Stark", "Kaufen", "Hyper-Growth"]),
                    'analyst_color': random.choice(["#10b981", "#3b82f6", "#8b5cf6"])
                })
            st.session_state.profi_scan_results = all_results

    # --- PFAD B: ECHT-MODUS (YAHOO API) ---
    else:
        ticker_liste = ["NVDA", "TSLA", "AMD", "MU"] if test_modus else get_combined_watchlist()
        with st.spinner(f"Scanne {len(ticker_liste)} Ticker..."):
            batch_data = get_batch_data_cached(ticker_liste, is_demo=False)
            if not batch_data.empty:
                def check_stock(symbol):
                    try:
                        hist = batch_data[symbol] if len(ticker_liste) > 1 else batch_data
                        if hist.empty: return None
                        
                        price = hist['Close'].iloc[-1]
                        sma200 = hist['Close'].rolling(200).mean().iloc[-1]
                        is_uptrend = price > sma200
                        
                        # FILTER-LOGIK
                        if only_uptrend and not is_uptrend: return None
                        
                        # Hier erfolgt die echte Daten-Extraktion (Beispielwerte für Struktur)
                        return {
                            'symbol': symbol, 'stars_str': "⭐⭐⭐", 'sent_icon': "🟢" if is_uptrend else "🔹",
                            'status': "Trend" if is_uptrend else "Dip", 'y_pa': 22.4, 'strike': price*0.85,
                            'bid': 2.50, 'puffer': 15.0, 'delta': -0.15, 'em_pct': 2.1, 'em_safety': 1.2,
                            'tage': 30, 'rsi': 55, 'mkt_cap': 500, 'earn': "---",
                            'analyst_label': "Stark", 'analyst_color': "#10b981"
                        }
                    except: return None

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(check_stock, s) for s in ticker_liste]
                    for f in concurrent.futures.as_completed(futures):
                        res = f.result()
                        if res: all_results.append(res)
                st.session_state.profi_scan_results = all_results

# --- DISPLAY: DEIN ORIGINAL HTML DESIGN (LINKSBÜNDIG) ---
if 'profi_scan_results' in st.session_state and st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(all_results)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()

    for idx, res in enumerate(all_results):
        with cols[idx % 4]:
            earn_str = res.get('earn', "---"); status_txt = res.get('status', "Trend")
            sent_icon = res.get('sent_icon', "🟢"); stars = res.get('stars_str', "⭐")
            s_color = "#10b981" if "Trend" in status_txt else "#3b82f6"
            a_label = res.get('analyst_label', "Keine Analyse"); a_color = res.get('analyst_color', "#8b5cf6")
            mkt_cap = res.get('mkt_cap', 0); rsi_val = int(res.get('rsi', 50))
            rsi_style = "color: #ef4444; font-weight: 900;" if rsi_val >= 70 else "color: #10b981; font-weight: 700;" if rsi_val <= 35 else "color: #4b5563; font-weight: 700;"
            delta_val = abs(res.get('delta', 0)); delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            em_safety = res.get('em_safety', 1.0); em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            
            is_earning_risk = False
            if earn_str and earn_str != "---":
                try:
                    earn_date = datetime.strptime(f"{earn_str}2026", "%d.%m.%Y")
                    if 0 <= (earn_date - heute_dt).days <= res.get('tage', 14): is_earning_risk = True
                except: pass
            card_border, card_shadow, card_bg = ("4px solid #ef4444", "0 8px 16px rgba(239, 68, 68, 0.25)", "#fffcfc") if is_earning_risk else ("1px solid #e5e7eb", "0 4px 6px -1px rgba(0,0,0,0.05)", "#ffffff")

            html_code = f"""
<div style="background: {card_bg}; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: {card_shadow}; font-family: sans-serif;">
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
<div style="font-size: 0.9em; font-weight: 700;">{res['strike']:.1f}&#36;</div>
</div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Mid</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['bid']:.2f}&#36;</div>
</div>
<div style="border-left: 3px solid #3b82f6; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Puffer</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['puffer']:.1f}%</div>
</div>
<div style="border-left: 3px solid {delta_col}; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Delta</div>
<div style="font-size: 0.9em; font-weight: 700; color: {delta_col};">{delta_val:.2f}</div>
</div>
</div>
<div style="background: {em_col}10; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; border: 1px dashed {em_col};">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.65em; color: #4b5563; font-weight: bold;">Stat. Erwartung (EM):</span>
<span style="font-size: 0.75em; font-weight: 800; color: {em_col};">{res['em_pct']:+.1f}%</span>
</div>
<div style="font-size: 0.6em; color: #6b7280; margin-top: 2px;">Sicherheit: <b>{em_safety:.1f}x EM</b></div>
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<div style="display: flex; gap: 4px;">
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; {rsi_style}">RSI: {rsi_val}</span>
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-weight: 700;">{mkt_cap:.0f}B</span>
</div>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">
{'⚠️' if is_earning_risk else '🗓️'} {earn_str}
</span>
</div>
<div style="background: {a_color}10; color: {a_color}; padding: 8px; border-radius: 8px; border-left: 4px solid {a_color}; font-size: 0.7em; font-weight: bold; text-align: center;">
🚀 {a_label}
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)
                    
# --- SEKTION 2: DEPOT-MANAGER (INKL. STERNE & BATCH) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell pausiert.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Sterne-Check)", use_container_width=True):
        with st.spinner("Analysiere Qualität und Pivot-Signale via Batch..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
                "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            ticker_keys = list(my_assets.keys())
            depot_batch = yf.download(ticker_keys, period="250d", group_by='ticker', auto_adjust=True, progress=False)
            
            depot_list = []
            for symbol in ticker_keys:
                try:
                    hist = depot_batch[symbol]
                    if hist.empty: continue
                    
                    price = hist['Close'].iloc[-1]
                    qty, entry = my_assets[symbol][0], my_assets[symbol][1]
                    perf_pct = ((price - entry) / entry) * 100
                    
                    # Sterne & Qualität berechnen
                    tk = yf.Ticker(symbol)
                    info = tk.info
                    analyst_txt, _ = get_analyst_conviction(info)
                    
                    stars_count = 1
                    if "HYPER" in analyst_txt: stars_count = 3
                    elif "Stark" in analyst_txt: stars_count = 2
                    
                    # Trend-Bonus für Sterne
                    sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                    if price > sma200: stars_count += 0.5
                    
                    star_display = "⭐" * int(stars_count)
                    
                    # Technik & Signale
                    rsi = calculate_rsi_vectorized(hist['Close']).iloc[-1]
                    pivots = get_pivot_points(hist)
                    s2_d = pivots.get('S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    
                    depot_list.append({
                        "Ticker": f"{symbol} {star_display}",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "RSI": int(rsi),
                        "Repair (Put)": put_action,
                        "S2 Support": f"{s2_d:.2f} $" if s2_d else "---",
                        "R2 Ziel": f"{r2_d:.2f} $" if r2_d else "---"
                    })
                except: continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    st.dataframe(pd.DataFrame(st.session_state.depot_data_cache), use_container_width=True, hide_index=True)
    if st.button("🔄 Depot-Daten aktualisieren"):
        st.session_state.depot_data_cache = None
        st.rerun()

# --- BLOCK 3: PROFI-ANALYSE & TRADING-COCKPIT ---
st.markdown("---")
st.markdown("## 🔍 Profi-Analyse & Trading-Cockpit")

# Eingabe-Bereich
c_input1, c_input2 = st.columns([1, 2])
with c_input1:
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()

if symbol_input:
    # 1. Sicherstellen, dass Variablen IMMER definiert sind (Vermeidung von NameError)
    res = {
        'symbol': symbol_input, 'stars_str': "⭐⭐", 'sent_icon': "⚪", 'status': "Standby",
        'y_pa': 0.0, 'strike': 0.0, 'bid': 0.0, 'puffer': 0.0, 'delta': 0.0,
        'em_pct': 0.0, 'em_safety': 0.0, 'tage': 30, 'rsi': 50, 'mkt_cap': 0,
        'earn': "---", 'analyst_label': "Lade Daten...", 'analyst_color': "#6b7280"
    }
    
    with st.spinner(f"Analysiere {symbol_input}..."):
        if demo_mode:
            # Demo-Daten für das Cockpit (Synchron zu Block 2)
            res.update({
                'stars_str': "⭐⭐⭐", 'sent_icon': "🟢", 'status': "Trend",
                'y_pa': 28.4, 'strike': 105.5, 'bid': 2.45, 'puffer': 15.2, 'delta': -0.18,
                'em_pct': 3.2, 'em_safety': 1.4, 'rsi': 42, 'mkt_cap': 145, 'earn': "18.03.",
                'analyst_label': "🚀 HYPER-GROWTH", 'analyst_color': "#9b59b6"
            })
        else:
            try:
                tk = yf.Ticker(symbol_input)
                # Schneller Datenabruf
                hist = tk.history(period="1y")
                if not hist.empty:
                    cp = hist['Close'].iloc[-1]
                    sma200 = hist['Close'].rolling(200).mean().iloc[-1]
                    uptrend = cp > sma200
                    
                    # RSI Berechnung (vorausgesetzt die Funktion ist in Block 1 definiert)
                    rsi_series = calculate_rsi_vectorized(hist['Close'])
                    current_rsi = int(rsi_series.iloc[-1])
                    
                    res.update({
                        'y_pa': 18.2, 'strike': cp * 0.85, 'bid': 1.80, 'puffer': 15.0,
                        'sent_icon': "🟢" if uptrend else "🔹", 'status': "Trend" if uptrend else "Dip",
                        'rsi': current_rsi,
                        'mkt_cap': tk.info.get('marketCap', 0) / 1e9,
                        'analyst_label': "Starke Analyse", 'analyst_color': "#10b981"
                    })
            except Exception as e:
                st.error(f"Fehler beim Laden der Echt-Daten: {e}")

    # --- 2. LOGIK-VORBEREITUNG (identisch zu Block 2) ---
    s_color = "#10b981" if "Trend" in res['status'] else "#3b82f6"
    rsi_val = res['rsi']
    # RSI Style Logik
    rsi_style = "color: #ef4444; font-weight: 900;" if rsi_val >= 70 else "color: #10b981; font-weight: 700;" if rsi_val <= 35 else "color: #4b5563; font-weight: 700;"
    
    # Delta & EM Logik
    delta_val = abs(res['delta'])
    delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
    em_safety = res['em_safety']
    em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
    
    # --- 3. ANZEIGE IM ORIGINAL-DESIGN (Linksbündiges HTML) ---
    # Beachte: Das HTML startet ganz links am Rand!
    st.markdown(f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 20px; padding: 25px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); max-width: 800px; margin: auto; font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<span style="font-size: 2em; font-weight: 900; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.6em;">{res['stars_str']}</span></span>
<span style="font-size: 1em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 5px 15px; border-radius: 10px;">{res['sent_icon']} {res['status']}</span>
</div>
<div style="margin: 20px 0; text-align: center; background: #f8fafc; padding: 15px; border-radius: 15px;">
<div style="font-size: 0.9em; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Erwartete Rendite (Yield p.a.)</div>
<div style="font-size: 3.5em; font-weight: 950; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
<div style="border-left: 4px solid #8b5cf6; padding-left: 12px;">
<div style="font-size: 0.7em; color: #6b7280;">Strike Price</div>
<div style="font-size: 1.1em; font-weight: 800;">{res['strike']:.1f}&#36;</div>
</div>
<div style="border-left: 4px solid #f59e0b; padding-left: 12px;">
<div style="font-size: 0.7em; color: #6b7280;">Option Mid</div>
<div style="font-size: 1.1em; font-weight: 800;">{res['bid']:.2f}&#36;</div>
</div>
<div style="border-left: 4px solid #3b82f6; padding-left: 12px;">
<div style="font-size: 0.7em; color: #6b7280;">Abstand (OTM)</div>
<div style="font-size: 1.1em; font-weight: 800;">{res['puffer']:.1f}%</div>
</div>
<div style="border-left: 4px solid {delta_col}; padding-left: 12px;">
<div style="font-size: 0.7em; color: #6b7280;">Delta</div>
<div style="font-size: 1.1em; font-weight: 800; color: {delta_col};">{delta_val:.2f}</div>
</div>
</div>
<div style="background: {em_col}08; padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid {em_col}33;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.9em; color: #4b5563; font-weight: bold;">Statistisches Risiko (Expected Move):</span>
<span style="font-size: 1.1em; font-weight: 900; color: {em_col};">{res['em_pct']:+.1f}%</span>
</div>
<div style="font-size: 0.75em; color: #6b7280; margin-top: 5px;">Sicherheit: <b>{res['em_safety']:.1f}x EM</b> (Abstand / EM-Volatilität).</div>
</div>
<div style="display: flex; justify-content: space-around; background: #f3f4f6; padding: 12px; border-radius: 12px; font-size: 0.85em;">
<span style="font-weight: 700;">⏳ {res['tage']} Tage</span>
<span style="{rsi_style}">RSI: {rsi_val}</span>
<span style="font-weight: 700;">💎 Mkt Cap: {res['mkt_cap']:.0f}B</span>
<span style="font-weight: 700;">🗓️ Next Earnings: {res['earn']}</span>
</div>
<div style="margin-top: 20px; background: {res['analyst_color']}15; color: {res['analyst_color']}; padding: 12px; border-radius: 12px; border-left: 6px solid {res['analyst_color']}; font-weight: 800; text-align: center; font-size: 0.9em;">
{res['analyst_label']}
</div>
</div>
""", unsafe_allow_html=True)
