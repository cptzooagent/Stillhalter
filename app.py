import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
import random
from datetime import datetime

# --- 1. MATHEMATISCHE HILFSFUNKTIONEN ---
def calculate_rsi_vectorized(series, window=14):
    """Berechnet den RSI effizient für eine ganze Datenserie."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Vermeidung von Division durch Null
    loss = loss.replace(0, 0.00001)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. S&P 500 WATCHLIST FUNKTION (STABIL) ---
@st.cache_data(ttl=86400)
def get_combined_watchlist():
    """
    Lädt Ticker von einer stabilen GitHub-Quelle (verhindert 403 Forbidden Fehler).
    Kombiniert diese mit dem User-Depot.
    """
    # Quelle: Offizieller S&P 500 Datensatz auf GitHub
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    
    try:
        df = pd.read_csv(url)
        # Yahoo Finance Korrektur: Punkte in Tickern durch Bindestriche ersetzen (z.B. BRK.B -> BRK-B)
        watchlist = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        # Fallback-Liste falls GitHub nicht erreichbar ist
        watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "NFLX", "DIS"]
        st.sidebar.warning(f"S&P 500 Download fehlgeschlagen, nutze Notfall-Liste.")
    
    # Depot-Integration (falls vorhanden)
    if 'depot_df' in st.session_state and not st.session_state['depot_df'].empty:
        try:
            # Wir extrahieren alle Symbole aus deinem Depot-DataFrame
            depot_symbols = st.session_state['depot_df']['Symbol'].dropna().unique().tolist()
            # Kombinieren und Duplikate entfernen
            watchlist = list(set(watchlist + depot_symbols))
        except:
            pass
            
    return watchlist

# --- 3. INITIALISIERUNG SESSION STATE ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

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

# --- BLOCK 2: PROFI-SCANNER ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 S&P 500 Profi-Scan starten", key="run_pro_scan_final", use_container_width=True):
    all_results = []
    
    if demo_mode:
        with st.spinner("Demo-Modus aktiv..."):
            time.sleep(1)
            for s in ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "PLTR"]:
                all_results.append({
                    'symbol': s, 'stars_str': "⭐⭐⭐", 'sent_icon': "🟢", 'status': "Trend",
                    'y_pa': 25.0, 'strike': 150.0, 'bid': 2.50, 'puffer': 15.0, 'delta': -0.15,
                    'em_pct': 4.0, 'em_safety': 1.5, 'tage': 30, 'rsi': 40, 'mkt_cap': 1500,
                    'analyst_label': "Demo", 'analyst_color': "#10b981"
                })
            st.session_state.profi_scan_results = all_results
    else:
        try:
            ticker_liste = get_combined_watchlist()
            if test_modus: # Variable aus deiner Sidebar
                ticker_liste = ticker_liste[:12]

            with st.spinner(f"📥 Batch-Download für {len(ticker_liste)} Ticker..."):
                all_data = yf.download(ticker_liste, period="1y", group_by='ticker', threads=True, progress=False)

            prog_bar = st.progress(0)
            status_txt = st.empty()

            for idx, symbol in enumerate(ticker_liste):
                status_txt.text(f"Analyse: {symbol}")
                prog_bar.progress((idx + 1) / len(ticker_liste))
                try:
                    df = all_data[symbol].dropna() if len(ticker_liste) > 1 else all_data.dropna()
                    if len(df) < 150: continue
                    
                    tk = yf.Ticker(symbol)
                    fi = tk.fast_info
                    cp = fi.last_price
                    mkt_cap = fi.market_cap / 1e9

                    if not (min_stock_price <= cp <= max_stock_price): continue
                    if mkt_cap < min_mkt_cap: continue

                    sma200 = df['Close'].rolling(200).mean().iloc[-1]
                    is_uptrend = cp > sma200
                    if only_uptrend and not is_uptrend: continue
                    
                    rsi_val = calculate_rsi_vectorized(df['Close']).iloc[-1]

                    # Individuelle Berechnung
                    log_ret = np.log(df['Close'] / df['Close'].shift(1))
                    vol = log_ret.std() * np.sqrt(252)
                    em = (vol * np.sqrt(30/365)) * 100
                    puffer = float(otm_puffer_slider)

                    all_results.append({
                        'symbol': symbol,
                        'stars_str': "⭐⭐⭐" if (is_uptrend and rsi_val < 45) else "⭐⭐",
                        'star_prio': (3 if (is_uptrend and rsi_val < 45) else 2) * 100 + (1 if is_uptrend else 0),
                        'sent_icon': "🟢" if is_uptrend else "🔹",
                        'status': "Trend" if is_uptrend else "Dip",
                        'y_pa': 10 + (vol * 35),
                        'strike': cp * (1 - puffer/100),
                        'bid': cp * (vol * 0.04),
                        'puffer': puffer,
                        'delta': -0.5 * (1 - (puffer/(em*2.5))),
                        'em_pct': em,
                        'em_safety': puffer/em if em > 0 else 1.0,
                        'tage': 30, 'rsi': int(rsi_val), 'mkt_cap': mkt_cap,
                        'analyst_label': "Uptrend" if is_uptrend else "Rebound",
                        'analyst_color': "#10b981" if is_uptrend else "#3498db"
                    })
                except: continue
            
            st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['star_prio'], reverse=True)
            prog_bar.empty()
            status_txt.empty()
        except Exception as e:
            st.error(f"Fehler: {e}")

# --- BLOCK 3: ANZEIGE ---
if st.session_state.profi_scan_results:
    st.write("---")
    cols = st.columns(4)
    for idx, res in enumerate(st.session_state.profi_scan_results):
        with cols[idx % 4]:
            s_color = "#10b981" if "Trend" in res['status'] else "#3b82f6"
            rsi_bg = "#dcfce7" if res['rsi'] <= 35 else "#f3f4f6"
            rsi_txt = "#166534" if res['rsi'] <= 35 else "#4b5563"
            
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
<div style="border-left: 3px solid #10b981; padding-left: 8px;">
<div style="font-size: 0.65em; color: #6b7280;">Delta</div>
<div style="font-size: 0.9em; font-weight: 700;">{abs(res['delta']):.2f}</div>
</div>
</div>
<div style="background: #f3f4f6; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; border: 1px dashed #d1d5db;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.65em; color: #4b5563; font-weight: bold;">Stat. Erwartung (EM):</span>
<span style="font-size: 0.75em; font-weight: 800;">{res['em_pct']:+.1f}%</span>
</div>
</div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<div style="display: flex; gap: 4px;">
<span style="background: {rsi_bg}; color: {rsi_txt}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">RSI: {res['rsi']}</span>
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

