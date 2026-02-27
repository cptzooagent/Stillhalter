import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import concurrent.futures
import time
import requests

# --- 1. SETUP & KONFIGURATION ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 2. MATHEMATISCHE FUNKTIONEN ---
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
        return {"P": p_d, "S1": s1_d, "S2": s2_d, "R2": r2_d, "W_S2": p_w - (h_w - l_w), "W_R2": p_w + (h_w - l_w)}
    except: return None

# --- 3. ANALYSE-FUNKTIONEN (WICHTIG: HIER DEFINIERT FÜR ALLE SEKTIONEN) ---
def get_openclaw_analysis(symbol):
    """KI-News Sentiment Analyse."""
    try:
        tk = yf.Ticker(symbol)
        all_news = tk.news
        if not all_news: return "Neutral", "🤖 Keine News-Daten.", 0.5
        
        huge_blob = str(all_news).lower()
        display_text = ""
        for n in all_news:
            for val in n.values():
                if isinstance(val, str) and val.count(" ") > 3:
                    display_text = val
                    break
            if display_text: break
        
        if not display_text: display_text = all_news[0].get('title', 'Markt aktiv')

        score = 0.5
        bull_words = ['earnings', 'growth', 'beat', 'buy', 'profit', 'ai', 'demand', 'up', 'upgrade']
        bear_words = ['sell-off', 'disruption', 'miss', 'down', 'risk', 'decline', 'warning', 'sell']
        for w in bull_words: 
            if w in huge_blob: score += 0.08
        for w in bear_words: 
            if w in huge_blob: score -= 0.08
            
        score = max(0.1, min(0.9, score))
        status = "Bullish" if score > 0.55 else "Bearish" if score < 0.45 else "Neutral"
        icon = "🟢" if status == "Bullish" else "🔴" if status == "Bearish" else "🟡"
        return status, f"{icon} OpenClaw: {display_text[:90]}", score
    except: return "N/A", "🤖 OpenClaw: System-Reset...", 0.5

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        if rev_growth > 40: return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}%)", "#9b59b6"
        elif upside > 15: return f"✅ Stark (Ziel: +{upside:.0f}%)", "#27ae60"
        elif upside > 25: return f"💎 Quality-Dip (+{upside:.0f}%)", "#2980b9"
        return f"⚖️ Neutral", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

@st.cache_data(ttl=3600)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="200d") 
        if hist.empty: return None, [], "", 50, True, False, 0, None
        price = hist['Close'].iloc[-1]
        dates = list(tk.options)
        rsi_val = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
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

# --- 4. SIDEBAR (Design wie gewünscht) ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    otm_puffer_slider = st.slider("Puffer (%)", 3, 25, 15, key="puffer_sid")
    min_yield_pa = st.number_input("Mindest-Yield %", 0, 100, 12, key="yield_sid")
    min_stock_price = st.number_input("Min. Preis ($)", 0, 1000, 60, key="min_p")
    max_stock_price = st.number_input("Max. Preis ($)", 0, 1000, 500, key="max_p")
    st.markdown("---")
    min_mkt_cap = st.slider("Mkt-Cap (Mrd. $)", 1, 1000, 20, key="mkt_sid")
    only_uptrend = st.checkbox("Nur Aufwärtstrend", value=False, key="trend_sid")
    test_modus = st.checkbox("🛠️ Simulations-Modus", value=True, key="sim_sid")

# --- 5. GLOBAL MONITORING ---
def get_market_data():
    try:
        ndq = yf.Ticker("^NDX")
        vix = yf.Ticker("^VIX")
        btc = yf.Ticker("BTC-USD")
        h_ndq = ndq.history(period="30d")
        cp_ndq = ndq.fast_info.last_price
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        rsi_ndq = calculate_rsi(h_ndq['Close']).iloc[-1]
        v_val = vix.fast_info.last_price
        b_val = btc.fast_info.last_price
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except: return 0, 50, 0, 20, 0

st.markdown("## 🌍 Globales Markt-Monitoring")
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()

# Banner Logik
if dist_ndq < -2 or vix_val > 25:
    m_color, m_text, m_advice = "#e74c3c", "🚨 MARKT-ALARM: Nasdaq Schwäche", "Defensiv agieren. Fokus auf Depot-Absicherung."
else:
    m_color, m_text, m_advice = "#27ae60", "✅ TRENDSTARK: Marktumfeld ist konstruktiv", "Puts auf starke Aktien bei Rücksetzern möglich."

st.markdown(f"""
    <div style="background-color: {m_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h3 style="margin:0; font-size: 1.6em;">{m_text}</h3>
        <p style="margin:5px 0 0 0; opacity: 0.9; font-size: 1.1em;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
m2.metric("Bitcoin", f"{btc_val:,.0f} $")
m3.metric("VIX (Angst)", f"{vix_val:.2f}", delta="Hoch" if vix_val > 22 else "Normal", delta_color="inverse")

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
        if test_modus:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR"]
        else:
            ticker_liste = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META"] 

        all_results = []
        heute = datetime.now()

        def check_single_stock(symbol):
            try:
                tk = yf.Ticker(symbol)
                f_info = tk.fast_info
                curr_price = f_info.get('last_price') or f_info.get('lastPrice')
                m_cap = f_info.get('market_cap') or f_info.get('marketCap')

                if not curr_price or m_cap < mkt_cap_limit: return None
                if not (min_stock_price <= curr_price <= max_stock_price): return None

                res = get_stock_data_full(symbol)
                if res[0] is None: return None
                price, dates, earn, rsi, uptrend, _, atr, pivots = res

                if only_uptrend and not uptrend: return None

                valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 24]
                if not valid_dates: return None
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - puffer_val)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                
                if not opts.empty:
                    o = opts.iloc[0]
                    bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                    days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                    y_pa = (bid_val / o['strike']) * (365 / max(1, days_to_exp)) * 100
                    
                    if y_pa >= min_yield_pa:
                        info = tk.info
                        analyst_txt, analyst_col = get_analyst_conviction(info)
                        
                        # Fix: Sterne-String generieren
                        stars_count = 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1
                        stars_str = "⭐" * stars_count
                        
                        # Alle Keys definieren, die unten gebraucht werden!
                        return {
                            'symbol': symbol, 
                            'price': price, 
                            'y_pa': y_pa, 
                            'strike': o['strike'], 
                            'bid': bid_val,
                            'tage': days_to_exp,
                            'puffer': ((price - o['strike']) / price) * 100,
                            'rsi': rsi, 
                            'earn': earn if earn else "n.a.", 
                            'status': "🛡️ Trend" if uptrend else "💎 Dip",
                            'analyst_txt': analyst_txt, 
                            'analyst_col': analyst_col,
                            'stars_val': stars_count,
                            'stars_str': stars_str
                        }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_single_stock, s) for s in ticker_liste]
            for future in concurrent.futures.as_completed(futures):
                res_data = future.result()
                if res_data: all_results.append(res_data)

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)

# --- RESULTATE ANZEIGEN ---
if st.session_state.profi_scan_results:
    all_results = st.session_state.profi_scan_results
    st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
    
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
                # UI-Kachel
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
                    
# --- 9. SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (INKL. PANIK-SCHUTZ) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            # info = tk.info # Hinweis: info wird unten für analyst_conviction gebraucht
            res = get_stock_data_full(symbol_input)

            if res[0] is not None:
                # Korrektes Entpacken der 8 Rückgabewerte
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res 
                
                # Analysten-Daten abrufen
                analyst_txt, analyst_col = get_analyst_conviction(tk.info)

                # --- 1. Earnings-Warnung ---
                if earn and earn != "---":
                    if "Feb" in earn or "Mar" in earn:
                        st.error(f"⚠️ **Earnings-Warnung:** Nächste Zahlen am {earn}. Vorsicht bei neuen Trades!")
                    else:
                        st.info(f"🗓️ Nächste Earnings: {earn}")
                else:
                    st.write("🗓️ Keine Earnings-Daten verfügbar")

                # --- 2. STRATEGIE-SIGNAL (S2 Logik) ---
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
        
                put_action_scanner = "⏳ Warten (Kein Signal)"
                signal_color = "gray"

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b" 
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60"

                # Signal-Box
                st.markdown(f"""
                    <div style="padding:10px; border-radius:10px; border: 2px solid {signal_color}; text-align:center; margin-bottom: 20px;">
                        <small style="color: gray;">Aktuelles Short Put Signal:</small><br>
                        <strong style="font-size:20px; color:{signal_color};">{put_action_scanner}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Sterne-Logik
                stars = 0
                if "HYPER" in analyst_txt: stars = 3
                elif "Stark" in analyst_txt: stars = 2
                elif "Neutral" in analyst_txt: stars = 1
                if uptrend and stars > 0: stars += 1
                
                # --- AMPEL-LOGIK (PANIK-SCHUTZ) ---
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif stars >= 2.5 and uptrend and 30 <= rsi <= 60:
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ANALYSTEN-WARNUNG"

                # Anzeige Ampel-Header
                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0; font-size: 1.8em;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Metriken-Board
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Kurs", f"{price:.2f} $")
                m_col2.metric("RSI (14)", f"{int(rsi)}", delta="PANIK" if rsi < 25 else None, delta_color="inverse")
                m_col3.metric("Phase", f"{'🛡️ Trend' if uptrend else '💎 Dip'}")
                m_col4.metric("Qualität", "⭐" * int(stars))

                # --- PIVOT ANALYSE ANZEIGE ---
                st.markdown("---")
                if pivots_res:
                    st.markdown("#### 🛡️ Technische Absicherung & Ziele (Pivots)")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Weekly S2", f"{pivots_res.get('W_S2', 0):.2f} $")
                    pc2.metric("Daily S2", f"{pivots_res.get('S2', 0):.2f} $")
                    pc3.metric("Pivot (P)", f"{pivots_res.get('P', 0):.2f} $")
                    pc4.metric("Daily R2", f"{pivots_res.get('R2', 0):.2f} $")
                    pc5.metric("Weekly R2", f"{pivots_res.get('W_R2', 0):.2f} $")
                    st.caption(f"💡 **CC-Tipp:** Covered Call am R2 Weekly ({pivots_res.get('W_R2', 0):.2f} $) ist statistisch optimal.")

                # Analysten Box
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                        <hr>
                        <span>📅 Nächste Earnings: <b>{earn if earn else 'n.a.'}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # --- OPTIONEN TABELLE ---
                st.markdown("---")
                st.markdown("### 🎯 Option-Chain Auswahl")
                option_mode = st.radio("Strategie wählen:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                heute = datetime.now()
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days

                    # KI-News Sentiment (OpenClaw)
                    ki_status, ki_text, ki_score = get_openclaw_analysis(symbol_input)
                    st.info(ki_text)

                    opt_chain = tk.option_chain(target_date)
                    chain = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    df_disp = chain[chain['openInterest'] > 20].copy()

                    if "Put" in option_mode:
                        df_disp = df_disp[df_disp['strike'] < price].copy()
                        df_disp['Puffer %'] = ((price - df_disp['strike']) / price) * 100
                        sort_order = False
                    else:
                        df_disp = df_disp[df_disp['strike'] > price].copy()
                        df_disp['Puffer %'] = ((df_disp['strike'] - price) / price) * 100
                        sort_order = True

                    df_disp['Yield p.a. %'] = (df_disp['bid'] / df_disp['strike']) * (365 / max(1, days_to_expiry)) * 100
                    df_disp = df_disp.sort_values('strike', ascending=sort_order)

                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 5 <= p < 10: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    st.dataframe(df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(12).style.apply(style_rows, axis=1).format({
                        'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $', 'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                    }), use_container_width=True)
            else:
                st.error("Daten konnten nicht geladen werden.")

    except Exception as e:
        st.error(f"Fehler bei {symbol_input}: {e}")









