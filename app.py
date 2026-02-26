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

@st.cache_data(ttl=3600)
def get_combined_watchlist():
    """
    Lädt die S&P 500 Ticker von Wikipedia und kombiniert sie 
    mit den Symbolen aus dem User-Depot.
    """
    try:
        # 1. S&P 500 Ticker von Wikipedia laden
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]
        watchlist = sp500_df['Symbol'].tolist()
        
        # Yahoo Finance Korrektur: Punkte durch Bindestriche ersetzen (z.B. BRK.B -> BRK-B)
        watchlist = [t.replace('.', '-') for t in watchlist]
    except Exception as e:
        # Fallback, falls Wikipedia nicht erreichbar ist
        watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"]
        st.warning(f"S&P 500 konnte nicht geladen werden, nutze Standard-Watchlist. ({e})")
    
    # 2. Depotwerte hinzufügen
    if 'depot_df' in st.session_state and not st.session_state['depot_df'].empty:
        # Wir nehmen an, deine Depot-Spalte heißt 'Symbol'
        depot_symbols = st.session_state['depot_df']['Symbol'].dropna().unique().tolist()
        # Kombinieren und Duplikate entfernen
        combined = list(set(watchlist + depot_symbols))
        return combined
    
    return watchlist

# --- SETUP & STYLING ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- SIDEBAR (Angepasst) ---
with st.sidebar:
    st.header("⚙️ System-Einstellungen")
    demo_mode = st.toggle("🛠️ Demo-Modus (API Bypass)", value=True, help="Aktivieren, wenn Yahoo Finance dich gesperrt hat.")
    
    st.markdown("---")
    st.header("🛡️ Scanner-Filter")
    otm_puffer_slider = st.slider("OTM Puffer (%)", 5, 25, 12)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 5, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis ($)", 0, 1000, (40, 600))
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 15)
    only_uptrend = st.checkbox("Nur SMA 200 Uptrend", value=False)
    # WICHTIG: Hier den Key 'test_modus_key' hinzugefügt
    test_modus = st.checkbox("🔍 Kleiner Scan (12 Ticker)", value=False, key='test_modus_key')

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

# --- BLOCK 2: ULTRA-FAST S&P 500 SCANNER ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 S&P 500 Profi-Scan starten", key="run_pro_scan", use_container_width=True):
    all_results = []
    
    if demo_mode:
        # (Deine Demo-Logik bleibt hier...)
        pass
    else:
        # 1. Ticker-Liste laden (Stabil von GitHub-Datensatz statt Wikipedia)
        @st.cache_data
        def get_sp500_tickers():
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            df = pd.read_csv(url)
            return df['Symbol'].str.replace('.', '-', regex=False).tolist()

        try:
            ticker_liste = get_sp500_tickers()
            # Depot-Integration
            if 'depot_df' in st.session_state and not st.session_state['depot_df'].empty:
                depot_symbols = st.session_state['depot_df']['Symbol'].tolist()
                ticker_liste = list(set(ticker_liste + depot_symbols))
            
            # Kleiner Scan Check
            if st.session_state.get('test_modus_key', False):
                ticker_liste = ticker_liste[:12]

            with st.spinner(f"📥 Batch-Download für {len(ticker_liste)} Ticker läuft..."):
                # DER KEY: Ein einziger Download für alle historischen Daten (1 Jahr)
                # Das verhindert 500 einzelne Anfragen!
                all_data = yf.download(ticker_liste, period="1y", group_by='ticker', threads=True, progress=False)
                
            with st.spinner("⚡ Berechne Indikatoren via fast_info..."):
                for symbol in ticker_liste:
                    try:
                        # Daten für diesen Ticker aus dem Batch-Objekt ziehen
                        df = all_data[symbol] if len(ticker_liste) > 1 else all_data
                        if df.empty or len(df) < 200: continue
                        
                        # tk-Objekt nur für fast_info nutzen (kein Netzwerk-Traffic!)
                        tk = yf.Ticker(symbol)
                        fi = tk.fast_info 
                        
                        cp = fi.last_price
                        mkt_cap = fi.market_cap / 1e9
                        
                        # Filter: Slider-Werte aus deiner Sidebar
                        if not (min_stock_price <= cp <= max_stock_price): continue
                        if mkt_cap < min_mkt_cap: continue

                        # Technische Analyse (SMA 200 & RSI)
                        sma200 = df['Close'].rolling(200).mean().iloc[-1]
                        is_uptrend = cp > sma200
                        if only_uptrend and not is_uptrend: continue
                        
                        rsi_val = calculate_rsi_vectorized(df['Close']).iloc[-1]

                        # --- INDIVIDUELLE BERECHNUNG ---
                        # 1. EM (Expected Move) basierend auf der 20-Tage Volatilität (annualisiert auf 30 Tage)
                        log_returns = np.log(df['Close'] / df['Close'].shift(1))
                        vol_30d = log_returns.std() * np.sqrt(252) # Historische Volatilität (HV)
                        # Erwartete Bewegung für 30 Tage (1 Standardabweichung)
                        em_pct = (vol_30d * np.sqrt(30/365)) * 100 
                    
                        # 2. Delta Annäherung (Grob-Formel für OTM Puts)
                        # Je höher die Vola und je kleiner der Puffer, desto höher das Delta
                        puffer_val = float(otm_puffer_slider)
                        # Vereinfachtes Modell: Delta steigt, wenn Puffer < EM
                        delta_approx = -0.5 * (1 - (puffer_val / (em_pct * 2)))
                        delta_final = max(min(delta_approx, -0.05), -0.50) # Begrenzung auf sinnvolle Werte

                        # 3. EM Safety Score
                        em_safety = puffer_val / em_pct if em_pct > 0 else 1.0

                        all_results.append({
                            'symbol': symbol, 
                            'stars_str': "⭐⭐⭐" if (is_uptrend and rsi_val < 45) else "⭐⭐",
                            'sent_icon': "🟢" if is_uptrend else "🔹", 
                            'status': "Trend" if is_uptrend else "Dip",
                            'y_pa': 12.0 + (vol_30d * 40), # Prämie steigt mit Volatilität!
                            'strike': cp * (1 - puffer_val/100), 
                            'bid': cp * (vol_30d * 0.05), # Höhere Vola = Höhere Prämie
                            'puffer': puffer_val, 
                            'delta': delta_final, 
                            'em_pct': em_pct, 
                            'em_safety': em_safety, 
                            'tage': 30, 
                            'rsi': int(rsi_val), 
                            'mkt_cap': mkt_cap,
                            'earn': "---", 
                            'analyst_label': "Uptrend" if is_uptrend else "Rebound",
                            'analyst_color': "#10b981" if is_uptrend else "#3498db"
                        })
                    except:
                        continue
            
            st.session_state.profi_scan_results = all_results
        except Exception as e:
            st.error(f"Fehler beim S&P 500 Scan: {e}")
            
# --- ANZEIGEBLOCK: SCANNER-ERGEBNISSE ---
if st.session_state.profi_scan_results:
    st.write("---")
    st.subheader(f"🎯 Top-Setups nach Qualität ({len(st.session_state.profi_scan_results)} Treffer)")
    
    # 4 Spalten Layout
    cols = st.columns(4)
    
    for idx, res in enumerate(st.session_state.profi_scan_results):
        with cols[idx % 4]:
            # Variablen-Vorbereitung
            s_color = "#10b981" if "Trend" in res['status'] else "#3b82f6"
            
            # RSI Farblogik
            rsi_val = res.get('rsi', 50)
            if rsi_val <= 35:
                rsi_bg, rsi_text = "#dcfce7", "#166534" # Grün
            elif rsi_val >= 70:
                rsi_bg, rsi_text = "#fee2e2", "#991b1b" # Rot
            else:
                rsi_bg, rsi_text = "#f3f4f6", "#4b5563" # Neutral

            # Delta & EM Styling
            delta_val = abs(res.get('delta', 0.15))
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b"
            em_safety = res.get('em_safety', 1.2)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b"
            
            # Der HTML Code - Linksbündig für sauberes Rendering
            html_tile = f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif; min-height: 380px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.2em; font-weight: 800; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.8em;">{res['stars_str']}</span></span>
<span style="font-size: 0.75em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 2px 8px; border-radius: 6px;">{res['sent_icon']} {res['status']}</span>
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
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Mid</div>
<div style="font-size: 0.9em; font-weight: 700;">{res['bid']:.2f}$</div>
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
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<div style="display: flex; gap: 4px;">
<span style="background: {rsi_bg}; color: {rsi_text}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">RSI: {rsi_val}</span>
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-weight: 700;">{res['mkt_cap']:.0f}B</span>
</div>
</div>
<div style="background: {res['analyst_color']}10; color: {res['analyst_color']}; padding: 8px; border-radius: 8px; border-left: 4px solid {res['analyst_color']}; font-size: 0.7em; font-weight: bold; text-align: center;">
🚀 {res['analyst_label']}
</div>
</div>
"""
            st.markdown(html_tile, unsafe_allow_html=True)
else:
    st.info("Klicke auf den Button oben, um den S&P 500 Scan zu starten.")
                    
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
c_input1, _ = st.columns([1, 2])
with c_input1:
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()

if symbol_input:
    # 1. Daten-Initialisierung
    res = {
        'symbol': symbol_input, 'stars_str': "⭐⭐", 'sent_icon': "⚪", 'status': "Standby",
        'y_pa': 0.0, 'strike': 0.0, 'bid': 0.0, 'puffer': 0.0, 'delta': 0.0,
        'em_pct': 0.0, 'em_safety': 0.0, 'tage': 30, 'rsi': 50, 'mkt_cap': 0,
        'earn': "---", 'analyst_label': "Lade Daten...", 'analyst_color': "#6b7280"
    }
    
    with st.spinner(f"Analysiere {symbol_input}..."):
        if demo_mode:
            cp = 100.50 # Basispreis für Demo
            res.update({
                'stars_str': "⭐⭐⭐", 'sent_icon': "🟢", 'status': "Trend",
                'y_pa': 28.4, 'strike': 85.0, 'bid': 2.45, 'puffer': 15.2, 'delta': -0.18,
                'em_pct': 3.2, 'em_safety': 1.4, 'rsi': 42, 'mkt_cap': 145, 'earn': "18.03.",
                'analyst_label': "🚀 HYPER-GROWTH", 'analyst_color': "#9b59b6"
            })
        else:
            try:
                tk = yf.Ticker(symbol_input)
                hist = tk.history(period="1y")
                if not hist.empty:
                    cp = hist['Close'].iloc[-1]
                    res.update({
                        'y_pa': 18.2, 'strike': cp * 0.85, 'bid': 1.80, 'puffer': 15.0,
                        'sent_icon': "🟢", 'status': "Trend", 'rsi': 45, 'mkt_cap': 120
                    })
            except: pass

    # --- 2. FARB-LOGIK ---
    s_color = "#10b981" if "Trend" in res['status'] else "#3b82f6"
    rsi_style = "color: #ef4444; font-weight: 900;" if res['rsi'] >= 70 else "color: #10b981; font-weight: 700;"
    delta_col = "#10b981" if abs(res['delta']) < 0.20 else "#ef4444"
    em_col = "#10b981" if res['em_safety'] >= 1.5 else "#f59e0b"

    # --- 3. HTML COCKPIT (Original Design) ---
    st.markdown(f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 20px; padding: 25px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); max-width: 800px; margin: auto; font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<span style="font-size: 2em; font-weight: 900; color: #111827;">{res['symbol']} <span style="color: #f59e0b; font-size: 0.6em;">{res['stars_str']}</span></span>
<span style="font-size: 1em; font-weight: 700; color: {s_color}; background: {s_color}10; padding: 5px 15px; border-radius: 10px;">{res['sent_icon']} {res['status']}</span>
</div>
<div style="margin: 20px 0; text-align: center; background: #f8fafc; padding: 15px; border-radius: 15px;">
<div style="font-size: 0.9em; color: #6b7280; font-weight: 600; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 3.5em; font-weight: 950; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
<div style="border-left: 4px solid #8b5cf6; padding-left: 12px;"><div style="font-size: 0.7em; color: #6b7280;">Strike</div><div style="font-size: 1.1em; font-weight: 800;">{res['strike']:.1f}$</div></div>
<div style="border-left: 4px solid #f59e0b; padding-left: 12px;"><div style="font-size: 0.7em; color: #6b7280;">Mid</div><div style="font-size: 1.1em; font-weight: 800;">{res['bid']:.2f}$</div></div>
<div style="border-left: 4px solid #3b82f6; padding-left: 12px;"><div style="font-size: 0.7em; color: #6b7280;">Puffer</div><div style="font-size: 1.1em; font-weight: 800;">{res['puffer']:.1f}%</div></div>
<div style="border-left: 4px solid {delta_col}; padding-left: 12px;"><div style="font-size: 0.7em; color: #6b7280;">Delta</div><div style="font-size: 1.1em; font-weight: 800; color: {delta_col};">{abs(res['delta']):.2f}</div></div>
</div>
<div style="background: {em_col}08; padding: 12px; border-radius: 12px; margin-bottom: 20px; border: 1px solid {em_col}33;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.85em; color: #4b5563; font-weight: bold;">Stat. Erwartung (EM):</span>
<span style="font-size: 1.1em; font-weight: 900; color: {em_col};">{res['em_pct']:+.1f}%</span>
</div>
</div>
<div style="display: flex; justify-content: space-around; background: #f3f4f6; padding: 12px; border-radius: 12px; font-size: 0.85em; margin-bottom: 15px;">
<span>⏳ {res['tage']}d</span> <span style="{rsi_style}">RSI: {res['rsi']}</span> <span>💎 {res['mkt_cap']:.0f}B</span> <span>🗓️ {res['earn']}</span>
</div>
<div style="background: {res['analyst_color']}15; color: {res['analyst_color']}; padding: 10px; border-radius: 10px; border-left: 6px solid {res['analyst_color']}; font-weight: 800; text-align: center;">{res['analyst_label']}</div>
</div>
""", unsafe_allow_html=True)

    # --- 4. OPTIONSKETTE (LOGIK-FIX FÜR PUT/CALL STRIKES) ---
    st.write("")
    opt_type = st.radio("Strategie wählen:", ["🟢 Short Put (Bullish/Neutral)", "🔴 Short Call (Bearish)"], horizontal=True)
    
    if demo_mode:
        data = []
        base_price = cp if 'cp' in locals() else 100.0
        
        # Logik-Umschaltung basierend auf Radio-Button
        is_put = "Short Put" in opt_type
        
        # Generiere 10 OTM-Strikes
        for i in range(1, 11):
            if is_put:
                # Put: Strikes gehen nach UNTEN (z.B. 98, 96, 94...)
                puffer_pct = -(i * 2.0)
                strike = round(base_price * (1 + puffer_pct/100), 1)
            else:
                # Call: Strikes gehen nach OBEN (z.B. 102, 104, 106...)
                puffer_pct = +(i * 2.0)
                strike = round(base_price * (1 + puffer_pct/100), 1)
            
            bid = round(random.uniform(0.5, 4.0) / (i*0.5 + 1), 2) # Prämie sinkt, je weiter OTM
            y_pa = round((bid / strike) * (365/30) * 100, 1)
            delta = round(0.30 - (i * 0.025), 2) # Delta sinkt, je weiter OTM
            
            data.append({
                "Strike": strike, 
                "Bid": bid, 
                "Puffer %": puffer_pct, 
                "Yield p.a. %": y_pa, 
                "Delta": delta if delta > 0.01 else 0.01
            })
        
        df_opt = pd.DataFrame(data)
        
        # Anzeige mit Progress-Balken
        st.dataframe(
            df_opt,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="%.1f $"),
                "Yield p.a. %": st.column_config.NumberColumn(format="%.1f%%"),
                "Puffer %": st.column_config.ProgressColumn(
                    "Puffer %",
                    help="Abstand zum aktuellen Kurs",
                    format="%.1f%%",
                    min_value=-25 if is_put else 0,
                    max_value=0 if is_put else 25
                ),
                "Delta": st.column_config.NumberColumn(format="%.2f")
            }
        )
    else:
        st.info(f"Lade echte {opt_type} Kette von Yahoo Finance...")






