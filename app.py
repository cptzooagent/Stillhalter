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

# --- BLOCK 2: PROFI-SCANNER ---
if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="run_pro_scan", use_container_width=True):
    all_results = []
    
    # 1. PFAD: DEMO-MODUS (AKTIV BEI SPERRE)
    if demo_mode:
        with st.spinner("Generiere Demo-Setups für UI-Test..."):
            time.sleep(1) 
            demo_tickers = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "MU", "PLTR", "AMZN", "META", "COIN", "MSTR", "NFLX"]
            for s in demo_tickers:
                y_pa = random.uniform(15.0, 35.0)
                puffer = random.uniform(10.0, 20.0)
                price = random.uniform(100, 950)
                uptrend = random.choice([True, False])
                label, color = ("🚀 HYPER-GROWTH", "#9b59b6") if random.random() > 0.7 else ("✅ Stark", "#27ae60")
                s_val, s_str = get_stars_logic(label, uptrend)
                
                all_results.append({
                    'symbol': s, 'price': price, 'y_pa': y_pa, 'strike': price * (1 - puffer/100),
                    'puffer': puffer, 'bid': random.uniform(2.0, 15.0), 'rsi': random.randint(35, 65),
                    'tage': 32, 'em_pct': random.uniform(-4.5, 4.5), 
                    'em_col': "#27ae60" if random.random() > 0.5 else "#e74c3c",
                    'status': "🛡️ Trend" if uptrend else "💎 Dip",
                    'stars_val': s_val, 'stars_str': s_str, 'analyst_label': label,
                    'analyst_color': color, 'mkt_cap': random.uniform(100, 3000)
                })
            st.session_state.profi_scan_results = all_results
            st.success("Demo-Scan abgeschlossen!")

    # 2. PFAD: ECHT-MODUS (YAHOO API)
    else:
        ticker_liste = ["NVDA", "TSLA", "AMD", "MU"] if test_modus else get_combined_watchlist()
        with st.spinner(f"Scanne {len(ticker_liste)} Ticker (Batch Mode)..."):
            batch_data = get_batch_data_cached(ticker_liste, is_demo=False)
            
            if not batch_data.empty:
                def check_stock(symbol):
                    try:
                        hist = batch_data[symbol] if len(ticker_liste) > 1 else batch_data
                        if hist.empty or len(hist) < 20: return None
                        
                        price = hist['Close'].iloc[-1]
                        sma200 = hist['Close'].rolling(200).mean().iloc[-1]
                        uptrend = price > sma200
                        
                        if only_uptrend and not uptrend: return None
                        
                        tk = yf.Ticker(symbol)
                        # Minimales Setup für Echt-Daten-Rückgabe (Erweiterbar)
                        return {
                            'symbol': symbol, 'price': price, 'y_pa': 18.5, 'strike': price * 0.88,
                            'puffer': 12.0, 'bid': 2.50, 'rsi': 55, 'tage': 30,
                            'em_pct': 0.0, 'em_col': "#7f8c8d", 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                            'stars_val': 2.0, 'stars_str': "⭐⭐", 'analyst_label': "✅ Stark",
                            'analyst_color': "#27ae60", 'mkt_cap': 500
                        }
                    except: return None

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(check_stock, s) for s in ticker_liste]
                    for f in concurrent.futures.as_completed(futures):
                        res = f.result()
                        if res: all_results.append(res)
                st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['stars_val'], reverse=True)

# --- 3. DISPLAY: KACHELN (LINKSBÜNDIGES HTML) ---
if st.session_state.profi_scan_results:
    st.markdown(f"### 🎯 Top-Setups nach Qualität ({len(st.session_state.profi_scan_results)} Treffer)")
    res_list = st.session_state.profi_scan_results
    cols = st.columns(4)
    
    for i, res in enumerate(res_list):
        with cols[i % 4]:
            st.markdown(f"""
<div style="background-color: white; padding: 20px; border-radius: 15px; border: 1px solid #e6e9ef; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; min-height: 280px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<h2 style="margin:0; color: #1f2937; font-size: 1.4em;">{res['symbol']}</h2>
<span style="font-size: 0.8em;">{res['stars_str']}</span>
</div>
<div style="margin-top: 5px;">
<span style="font-size: 0.75em; font-weight: 800; color: {res['em_col']}; background: {res['em_col']}22; padding: 2px 6px; border-radius: 4px;">
{res['em_pct']:+.1f}%
</span>
</div>
<p style="color: {res['analyst_color']}; font-weight: bold; font-size: 0.85em; margin: 12px 0 5px 0;">{res['analyst_label']}</p>
<hr style="border: 0; border-top: 1px solid #eee; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
<span style="font-size: 0.85em; color: #6b7280;">Rendite p.a.</span>
<span style="font-size: 1em; font-weight: bold; color: #10b981;">{res['y_pa']:.1f}%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
<span style="font-size: 0.85em; color: #6b7280;">Strike (OTM)</span>
<span style="font-size: 0.9em; font-weight: 600;">{res['strike']:.1f}$ ({res['puffer']:.1f}%)</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
<span style="font-size: 0.85em; color: #6b7280;">Laufzeit</span>
<span style="font-size: 0.9em; font-weight: 600;">{res['tage']} Tage</span>
</div>
<div style="background: #f9fafb; padding: 8px; border-radius: 8px; display: flex; justify-content: space-around; font-size: 0.75em; font-weight: 600; color: #4b5563;">
<span>RSI: {int(res['rsi'])}</span>
<span>|</span>
<span>{res['status']}</span>
<span>|</span>
<span>{res['mkt_cap']:.0f}B</span>
</div>
</div>
""", unsafe_allow_html=True)
                    
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

# --- SEKTION 3: EINZEL-TICKER COCKPIT (KORRIGIERTER BUG) ---
st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", key="cockpit_input").upper()

if symbol_input:
    try:
        with st.spinner(f"Lade Cockpit für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            hist = tk.history(period="250d")
            if hist.empty:
                st.error("Keine historischen Daten gefunden.")
            else:
                price = hist['Close'].iloc[-1]
                
                # Technik
                rsi = calculate_rsi_vectorized(hist['Close']).iloc[-1]
                pivots = get_pivot_points(hist)
                sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                uptrend = price > sma200
                
                # Qualität & Sterne
                info = tk.info
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                stars_val = 1
                if "HYPER" in analyst_txt: stars_val = 3
                elif "Stark" in analyst_txt: stars_val = 2
                if uptrend: stars_val += 1
                star_display = "⭐" * min(int(stars_val), 4)

                # Ampel-Logik (KORREKTUR: ampel_color einheitlich)
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                if rsi < 25: 
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF"
                elif rsi > 75: 
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT"
                elif stars_val >= 3 and uptrend and 30 <= rsi <= 60: 
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"

                # Anzeige Ampel
                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0; font-size: 1.8em;">{star_display} {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Metriken
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kurs", f"{price:.2f} $")
                m2.metric("Qualität", star_display)
                m3.metric("RSI (14)", f"{int(rsi)}", delta="BUY" if rsi < 35 else "SELL" if rsi > 70 else None)
                m4.metric("Trend-Basis", "SMA 200 ↑" if uptrend else "SMA 200 ↓")

                # KI-Box
                ki_status, ki_text, _ = get_openclaw_analysis(symbol_input)
                st.info(ki_text)

                # Analysten Box
                st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Option-Chain
                st.markdown("---")
                st.subheader("🎯 Option-Chain Auswahl")
                opt_mode = st.radio("Strategie:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True, key="strat_radio")
                
                dates = tk.options
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 45]
                
                if valid_dates:
                    target_date = st.selectbox("Laufzeit wählen", valid_dates, key="date_select")
                    days = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                    
                    chain = tk.option_chain(target_date).puts if "Put" in opt_mode else tk.option_chain(target_date).calls
                    df_opt = chain[chain['openInterest'] > 10].copy()
                    
                    if not df_opt.empty:
                        if "Put" in opt_mode:
                            df_opt = df_opt[df_opt['strike'] < price].sort_values('strike', ascending=False)
                            df_opt['Puffer %'] = ((price - df_opt['strike']) / price) * 100
                        else:
                            df_opt = df_opt[df_opt['strike'] > price].sort_values('strike', ascending=True)
                            df_opt['Puffer %'] = ((df_opt['strike'] - price) / price) * 100
                        
                        df_opt['Yield p.a. %'] = (df_opt['bid'] / df_opt['strike']) * (365 / max(1, days)) * 100
                        
                        def style_rows(row):
                            if row['Puffer %'] >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                            return [''] * len(row)

                        st.dataframe(df_opt[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(10).style.apply(style_rows, axis=1).format({
                            'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $', 'Puffer %': '{:.1f}%', 'Yield p.a. %': '{:.1f}%'
                        }), use_container_width=True)
                    else:
                        st.warning("Keine liquiden Optionen gefunden.")

    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')} | © 2026 CapTrader AI")











