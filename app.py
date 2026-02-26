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

st.write("---")
st.header("🔍 Einzel-Ticker Quick-Check")

col_input, col_btn = st.columns([3, 1])
with col_input:
    manual_ticker = st.text_input("Ticker Symbol eingeben (z.B. MSFT, RHM.DE)", "").upper()

if col_btn.button("Einzel-Scan", use_container_width=True) and manual_ticker:
    with st.spinner(f"Analysiere {manual_ticker}..."):
        try:
            tk = yf.Ticker(manual_ticker)
            hist = tk.history(period="1y")
            if not hist.empty:
                cp = tk.fast_info.last_price
                rsi_single = calculate_rsi_vectorized(hist['Close']).iloc[-1]
                sma200_single = hist['Close'].rolling(200).mean().iloc[-1]
                
                # Kurze Info-Box
                st.success(f"**{manual_ticker}**: {cp:.2f}$ | RSI: {int(rsi_single)} | Trend: {'🟢 Aufwärts' if cp > sma200_single else '🔴 Abwärts'}")
                
                # Optional: Hier könnte man das gleiche Kachel-HTML wie in Block 3 anzeigen
            else:
                st.error("Keine Daten gefunden.")
        except Exception as e:
            st.error(f"Fehler beim Laden von {manual_ticker}: {e}")

st.write("---")
st.header("💼 Depot-Manager & Bestandsverwaltung")

# Initialisierung des Depot-Speichers
if 'depot_df' not in st.session_state:
    st.session_state.depot_df = pd.DataFrame(columns=['Symbol', 'Kaufkurs', 'Menge'])

# Eingabemaske
with st.expander("➕ Neue Position hinzufügen", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1: new_sym = st.text_input("Symbol", key="depot_sym").upper()
    with c2: new_price = st.number_input("Kaufpreis ($)", min_value=0.0, step=0.1)
    with c3: new_qty = st.number_input("Menge", min_value=0, step=1)
    
    if st.button("Position speichern"):
        if new_sym:
            new_row = pd.DataFrame([{'Symbol': new_sym, 'Kaufkurs': new_price, 'Menge': new_qty}])
            st.session_state.depot_df = pd.concat([st.session_state.depot_df, new_row], ignore_index=True)
            st.rerun()

# Anzeige der Tabelle
if not st.session_state.depot_df.empty:
    edited_df = st.data_editor(
        st.session_state.depot_df, 
        num_rows="dynamic", 
        use_container_width=True,
        key="depot_editor"
    )
    st.session_state.depot_df = edited_df
    
    if st.button("🗑️ Alle Bestände löschen"):
        st.session_state.depot_df = pd.DataFrame(columns=['Symbol', 'Kaufkurs', 'Menge'])
        st.rerun()
else:
    st.info("Dein Depot ist aktuell leer. Füge Symbole hinzu, um sie priorisiert zu scannen.")
