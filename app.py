import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
from curl_cffi import requests  # Die stabile Engine gegen Bot-Blocking

# --- 1. GRUNDKONFIGURATION ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 2. CURL_CFFI ENGINE (DIE BASIS) ---
def fetch_with_curl(url):
    """Holt Daten via curl_cffi im Chrome-Modus."""
    try:
        with requests.Session(impersonate="chrome110") as s:
            res = s.get(url, timeout=10)
            return res
    except Exception as e:
        return None

# --- 3. MATHEMATISCHE FUNKTIONEN ---
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
        tk = yf.Ticker(symbol)
        hist_d = tk.history(period="5d") 
        if len(hist_d) < 2: return None
        last_day = hist_d.iloc[-2]
        h_d, l_d, c_d = last_day['High'], last_day['Low'], last_day['Close']
        p_d = (h_d + l_d + c_d) / 3
        s1_d, s2_d, r2_d = (2 * p_d) - h_d, p_d - (h_d - l_d), p_d + (h_d - l_d)

        hist_w = tk.history(period="3wk", interval="1wk")
        if len(hist_w) < 2:
            return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_d, "W_R2": r2_d}
        
        last_week = hist_w.iloc[-2]
        h_w, l_w, c_w = last_week['High'], last_week['Low'], last_week['Close']
        p_w = (h_w + l_w + c_w) / 3
        s2_w, r2_w = p_w - (h_w - l_w), p_w + (h_w - l_w)
        return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": s2_w, "W_R2": r2_w}
    except: return None

# --- 4. DATEN-ANALYSEN ---
def get_openclaw_analysis(symbol):
    try:
        tk = yf.Ticker(symbol)
        all_news = tk.news
        if not all_news: return "Neutral", "🤖 Keine Daten.", 0.5
        huge_blob = str(all_news).lower()
        display_text = all_news[0].get('title', 'Markt aktiv')
        score = 0.5
        for w in ['growth', 'beat', 'buy', 'ai', 'profit']: 
            if w in huge_blob: score += 0.08
        for w in ['miss', 'risk', 'sell', 'warning', 'decline']: 
            if w in huge_blob: score -= 0.08
        score = max(0.1, min(0.9, score))
        status = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
        icon = "🟢" if status == "Bullish" else "🔴" if status == "Bearish" else "🟡"
        return status, f"{icon} OpenClaw: {display_text[:85]}...", score
    except: return "N/A", "🤖 System-Reset...", 0.5

@st.cache_data(ttl=3600)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="250d") 
        if hist.empty: return None, [], "", 50, True, False, 0, None
        price = hist['Close'].iloc[-1]
        dates = list(tk.options)
        rsi_val = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
        pivots = calculate_pivots(symbol)
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_str, rsi_val, (price > sma_200), False, atr, pivots
    except: return None, [], "", 50, True, False, 0, None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}%)", "#9b59b6"
        elif upside > 15: return f"✅ Stark (Ziel: +{upside:.0f}%)", "#27ae60"
        return f"⚖️ Neutral", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

# --- 5. SIDEBAR INITIALISIERUNG (WICHTIG: VOR DEM SCANNER!) ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    otm_puffer_slider = st.slider("Puffer (%)", 3, 25, 15)
    min_yield_pa = st.number_input("Mindest-Yield %", 0, 100, 12)
    min_stock_price = st.number_input("Min. Preis ($)", 0, 1000, 60)
    max_stock_price = st.number_input("Max. Preis ($)", 0, 1000, 500)
    min_mkt_cap = st.slider("Mkt-Cap (Mrd. $)", 1, 1000, 20)
    only_uptrend = st.checkbox("Nur Aufwärtstrend", value=False)
    test_modus = st.checkbox("🛠️ Simulations-Modus", value=True)

# --- 6. GLOBAL MONITORING (WIRD NUR EINMAL ANGEZEIGT) ---
def get_crypto_fg():
    res = fetch_with_curl("https://api.alternative.me/fng/")
    if res and res.status_code == 200:
        try: return int(res.json()['data'][0]['value'])
        except: return 50
    return 50

st.markdown("## 🌍 Globales Markt-Monitoring")
fg_val = get_crypto_fg()

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Fear & Greed (Crypto)", fg_val)
with m_col2:
    st.metric("Markt-Status", "Aktiv")
with m_col3:
    st.metric("Scanner-Engine", "curl_cffi / Chrome110")

st.markdown("---")

# --- 3. SEKTION 1: PROFI-SCANNER (ULTRA-SPEED) ---
st.header("🚀 Profi-Scanner (High Speed)")

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

# Sidebar-Werte abgreifen
puffer_val = otm_puffer_slider / 100
mkt_cap_limit = min_mkt_cap * 1_000_000_000

if st.button("🔍 Scan jetzt starten"):
    with st.spinner("Analysiere Markt mit parallelen Threads..."):
        # Ticker Liste bestimmen
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR"]
        else:
            # Hier käme deine get_combined_watchlist() zum Einsatz
            ticker_liste = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"] 

        all_results = []
        heute = datetime.now()

        def check_single_stock(symbol):
            try:
                tk = yf.Ticker(symbol)
                # Fast_Info für Vor-Filterung (Mkt Cap & Preis)
                f_info = tk.fast_info
                curr_price = f_info.get('last_price') or f_info.get('lastPrice')
                m_cap = f_info.get('market_cap') or f_info.get('marketCap')

                if not curr_price or m_cap < mkt_cap_limit: return None
                if not (min_stock_price <= curr_price <= max_stock_price): return None

                # Volle Analyse (RSI, SMA, Pivots)
                res = get_stock_data_full(symbol)
                if res[0] is None: return None
                price, dates, earn, rsi, uptrend, _, atr, pivots = res

                if only_uptrend and not uptrend: return None

                # Options-Check (Laufzeit 11-24 Tage)
                valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 24]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - puffer_val)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                
                if not opts.empty:
                    o = opts.iloc[0]
                    bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                    days = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                    y_pa = (bid_val / o['strike']) * (365 / max(1, days)) * 100
                    
                    if y_pa >= min_yield_pa:
                        info = tk.info
                        analyst_txt, analyst_col = get_analyst_conviction(info)
                        
                        return {
                            'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                            'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                            'rsi': rsi, 'earn': earn if earn else "n.a.", 
                            'status': "🛡️ Trend" if uptrend else "💎 Dip",
                            'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                            'stars_val': 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1
                        }
            except: return None

        # Parallelisierung (moderate Worker-Anzahl für Stabilität)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_single_stock, s) for s in ticker_liste]
            for future in concurrent.futures.as_completed(futures):
                res_data = future.result()
                if res_data: all_results.append(res_data)

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)

          

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

# DER BUTTON ALS AUSLÖSER
if st.button("🚀 Depot-Daten jetzt laden/aktualisieren", key="depot_load_btn"):
    
    # Deine Asset-Liste
    my_assets = {
        "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
        "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
        "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
        "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
    }

    with st.spinner('Lade Depot-Daten und berechne Signale via curl_cffi...'):
        with st.expander("📂 Mein Depot & Strategie-Signale", expanded=True):
            depot_list = []
            
            for symbol, data in my_assets.items():
                try:
                    # 1. Zentrale Daten abrufen
                    res = get_stock_data_full(symbol)
                    if res[0] is None: continue
                    
                    price, dates, earn, rsi, uptrend, _, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # 2. KI-Stimmung & Analysten (nur bei Bedarf für Depot-Check)
                    ki_status, ki_text, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
                    
                    # Schnelles Sterne-Rating (basiert auf Uptrend & RSI)
                    stars_count = 1
                    if uptrend: stars_count += 1
                    if 40 < rsi < 60: stars_count += 1
                    star_display = "⭐" * stars_count

                    # 3. Pivot-Werte extrahieren
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    
                    # 4. Reparatur-Logik (Short Put)
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02):
                        put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01:
                        put_action = "🔥 EXTREM (Weekly S2)"
                    
                    # 5. Covered Call Logik
                    call_action = "⏳ Warten"
                    if rsi > 55:
                        if r2_d and price >= r2_d * 0.98:
                            call_action = "🟢 JETZT (R2/RSI)"
                        else:
                            call_action = "🟡 RSI HOCH"

                    depot_list.append({
                        "Ticker": f"{symbol} {star_display}",
                        "Earnings": earn if earn else "---",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "KI-Check": f"{ki_icon} {ki_status}",
                        "RSI": int(rsi),
                        "Short Put (Repair)": put_action,
                        "Covered Call": call_action
                    })
                except Exception as e:
                    continue
            
            if depot_list:
                st.table(pd.DataFrame(depot_list))
                st.success(f"✅ {len(depot_list)} Positionen erfolgreich analysiert.")
            else:
                st.warning("Konnte keine Depot-Daten abrufen. Bitte API-Verbindung prüfen.")

    st.info("💡 **Tipp:** '🔥 EXTREM' bedeutet, die Aktie notiert am Weekly S2 – statistisch ein sehr starker Boden.")
                    
# --- SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (INKL. PANIK-SCHUTZ) ---
st.markdown("---")
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)

            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
                analyst_txt, analyst_col = get_analyst_conviction(info)

                # 1. EARNINGS-WARNUNG
                if earn and earn != "---":
                    # Warnung falls Earnings im aktuellen Zeitraum (Beispiel-Logik)
                    if "Feb" in earn or "Mar" in earn:
                        st.error(f"⚠️ **Earnings-Warnung:** Nächste Zahlen am {earn}. Erhöhte Volatilität erwartet!")
                    else:
                        st.info(f"🗓️ Nächste Earnings: {earn}")
                
                # 2. SIGNAL-BOX
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
                
                put_signal = "⏳ Warten"
                sig_col = "#7f8c8d"
                
                if s2_w and price <= s2_w * 1.01:
                    put_signal = "🔥 EXTREM (Weekly S2)"
                    sig_col = "#ff4b4b"
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_signal = "🟢 JETZT (S2/RSI)"
                    sig_col = "#27ae60"

                st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; border: 2px solid {sig_col}; text-align:center; margin-bottom:15px;">
                        <small>Strategie-Signal:</small><br>
                        <strong style="font-size:22px; color:{sig_col};">{put_signal}</strong>
                    </div>
                """, unsafe_allow_html=True)

                # 3. DIE SICHERHEITS-AMPEL
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                if rsi < 25:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif uptrend and 30 <= rsi <= 60:
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Trendstark)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: FUNDAMENTAL-WARNUNG"

                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # 4. METRIKEN
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Kurs", f"{price:.2f} $")
                c2.metric("RSI (14)", f"{int(rsi)}", delta="Tief" if rsi < 30 else None)
                c3.metric("Trend", "🛡️ Bull" if uptrend else "💎 Dip")
                c4.metric("ATR", f"{atr:.2f} $")

                # 5. OPTIONS-WAHL
                st.markdown("### 🎯 Option-Chain Auswahl")
                opt_mode = st.radio("Modus:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                heute = datetime.now()
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                
                if valid_dates:
                    t_date = st.selectbox("📅 Verfallstag wählen", valid_dates)
                    days_to_exp = (datetime.strptime(t_date, '%Y-%m-%d') - heute).days
                    
                    # OpenClaw KI-Snippet
                    _, ki_text, _ = get_openclaw_analysis(symbol_input)
                    st.caption(ki_text)

                    # Chain laden
                    opt_chain = tk.option_chain(t_date)
                    df_opt = opt_chain.puts if "Put" in opt_mode else opt_chain.calls
                    
                    # Filtern & Berechnen
                    if "Put" in opt_mode:
                        df_opt = df_opt[df_opt['strike'] < price].copy()
                        df_opt['Puffer %'] = ((price - df_opt['strike']) / price) * 100
                        sort_asc = False
                    else:
                        df_opt = df_opt[df_opt['strike'] > price].copy()
                        df_opt['Puffer %'] = ((df_opt['strike'] - price) / price) * 100
                        sort_asc = True
                    
                    df_opt['Yield p.a. %'] = (df_opt['bid'] / df_opt['strike']) * (365 / max(1, days_to_exp)) * 100
                    
                    # Anzeige
                    disp_df = df_opt[['strike', 'bid', 'ask', 'openInterest', 'Puffer %', 'Yield p.a. %']].head(12)
                    st.dataframe(disp_df.sort_values('strike', ascending=sort_asc), use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei Analyse {symbol_input}: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Engine: curl_cffi v1.0")




