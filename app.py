import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time
from requests import Session

# --- 1. ABSOLUTE INITIALISIERUNG (Gegen AttributeError) ---
if 'secure_session' not in st.session_state:
    session = Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    st.session_state.secure_session = session

def get_tk(symbol):
    """Sicherer Ticker-Aufruf."""
    return yf.Ticker(symbol, session=st.session_state.secure_session)

# --- 2. HILFSFUNKTIONEN ---
def calculate_rsi(data, window=14):
    if len(data) < window + 1: return pd.Series([50] * len(data))
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    loss = loss.replace(0, 0.001)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_pivots(symbol, batch_hist=None):
    try:
        hist = batch_hist if batch_hist is not None else get_tk(symbol).history(period="5d")
        if hist.empty or len(hist) < 2: return None
        last_day = hist.iloc[-2]
        h, l, c = last_day['High'], last_day['Low'], last_day['Close']
        p = (h + l + c) / 3
        return {"P": p, "S1": 2*p-h, "S2": p-(h-l), "R2": p+(h-l), "W_S2": (p-(h-l))*0.98, "W_R2": (p+(h-l))*1.02}
    except: return None

# --- 3. DIE REPARATUR FÜR DEIN COCKPIT (Gegen NameError) ---
def get_stock_data_full(symbol):
    """Wird vom Cockpit und Depot-Manager benötigt."""
    try:
        tk = get_tk(symbol)
        hist = tk.history(period="150d")
        if hist.empty: return None
        
        price = hist['Close'].iloc[-1]
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
        uptrend = price > sma_200
        
        # ATR Berechnung
        tr = pd.concat([hist['High']-hist['Low'], np.abs(hist['High']-hist['Close'].shift()), np.abs(hist['Low']-hist['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Earnings
        earn = "---"
        try:
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earn = cal.iloc[0, 0].strftime('%d.%m.')
        except: pass
        
        pivots = calculate_pivots(symbol, batch_hist=hist)
        return price, tk.options, earn, rsi, uptrend, False, atr, pivots
    except:
        return None

def get_analyst_conviction(info):
    try:
        current = info.get('currentPrice', 1)
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        if upside > 15: return f"✅ Stark (+{upside:.0f}%)", "#27ae60"
        elif upside < 0: return f"⚠️ Warnung ({upside:.1f}%)", "#e67e22"
        return f"⚖️ Neutral ({upside:.0f}%)", "#7f8c8d"
    except: return "🔍 Check nötig", "#7f8c8d"

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def get_openclaw_analysis(symbol):
    return "Neutral", "🤖 Standby", 0.5

@st.cache_data(ttl=86400)
def get_combined_watchlist():
    return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "MU", "AVGO", "PLTR", "APP", "NET", "CRWD"]

# --- APP START ---
st.set_page_config(page_title="CapTrader AI", layout="wide")

# Sidebar
with st.sidebar:
    st.header("🛡️ Filter")
    otm_puffer_slider = st.slider("Puffer (%)", 3, 25, 15)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12)
    min_stock_price, max_stock_price = st.slider("Preis ($)", 0, 1000, (40, 600))
    min_mkt_cap = st.slider("Market Cap (Mrd. $)", 1, 1000, 15)
    only_uptrend = st.checkbox("Nur Aufwärtstrend", value=False)
    test_modus = st.checkbox("🛠️ Simulation", value=True)

# Markt-Monitor (Fix für 'NoneType' Error)
st.markdown("## 🌍 Markt-Monitor")
col_m1, col_m2 = st.columns(2)
try:
    vix_tk = get_tk("^VIX")
    vix = vix_tk.history(period="1d")['Close'].iloc[-1]
    ndq_tk = get_tk("^NDX")
    ndq = ndq_tk.history(period="1d")['Close'].iloc[-1]
    col_m1.metric("Nasdaq 100", f"{ndq:,.0f}")
    col_m2.metric("VIX (Angst)", f"{vix:.2f}", delta="STABIL" if vix < 20 else "VOLATIL", delta_color="inverse")
except:
    st.info("Marktdaten aktuell verzögert (Wochenende/Pause).")
st.markdown("---")

# --- SEKTION 2: PROFI-SCANNER (BATCH-OPTIMIERT & STABILISIERT) ---

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
    
    with st.spinner(f"Batch-Analyse von {len(ticker_liste)} Symbolen läuft..."):
        # 1. SCHRITT: BATCH-DOWNLOAD (Alle Kurse & Historien auf einmal)
        # Dies ist der entscheidende Teil für die Stabilität
        batch_data = yf.download(
            ticker_liste, 
            period="150d", 
            group_by='ticker', 
            session=st.session_state.secure_session, 
            progress=False
        )
        
        all_results = []
        progress_bar = st.progress(0)

        # 2. SCHRITT: Verarbeitung der Daten
        for i, symbol in enumerate(ticker_liste):
            try:
                # Daten aus dem Batch extrahieren
                if symbol not in batch_data: continue
                hist = batch_data[symbol].dropna()
                if hist.empty: continue
                
                # Technische Basis-Checks (Lokal berechnet, kein API Call!)
                price = hist['Close'].iloc[-1]
                rsi_series = calculate_rsi(hist['Close'])
                rsi_val = rsi_series.iloc[-1]
                sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
                uptrend = price > sma_200
                
                if only_uptrend and not uptrend: continue
                if not (min_stock_price <= price <= max_stock_price): continue

                # 3. SCHRITT: Gezielte Abfrage von Info & Optionen
                tk = get_tk(symbol)
                info = tk.info
                
                # Market Cap Check
                m_cap = info.get('marketCap', 0)
                if m_cap < p_min_cap: continue
                
                # Options-Chain Analyse
                dates = tk.options
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                if not valid_dates: continue
                
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                
                opts = chain[(chain['strike'] <= target_strike) & (chain['openInterest'] > 1)].sort_values('strike', ascending=False)
                if opts.empty: continue
                o = opts.iloc[0]

                # Kalkulationen
                fair_price = (o['bid'] + o['ask']) / 2 if o['bid'] > 0 else o['lastPrice']
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (fair_price / o['strike']) * (365 / max(1, days_to_exp)) * 100
                
                if y_pa > 150 or y_pa < p_min_yield: continue

                # Risiko-Metriken
                iv = o.get('impliedVolatility', 0.4)
                exp_move_pct = (iv * np.sqrt(days_to_exp / 365)) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
                delta_val = calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv)
                
                # Earnings & Analysten
                earn = ""
                try:
                    cal = tk.calendar
                    if cal is not None and not cal.empty:
                        earn = cal.iloc[0, 0].strftime('%d.%m.')
                except: earn = "---"
                
                analyst_txt, analyst_col = get_analyst_conviction(info)
                stars_count = 3 if "HYPER" in analyst_txt else 2 if "Stark" in analyst_txt else 1

                all_results.append({
                    'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                    'puffer': current_puffer, 'bid': fair_price, 'rsi': rsi_val, 'earn': earn, 
                    'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 
                    'delta': abs(delta_val), 'sent_icon': "🟢" if uptrend else "🟡", 
                    'stars_str': "⭐" * stars_count, 
                    'analyst_label': analyst_txt, 'analyst_color': analyst_col, 
                    'mkt_cap': m_cap / 1e9, 'em_pct': exp_move_pct, 'em_safety': em_safety
                })
                
                # Kleine Pause zwischen den Symbolen für die restlichen API-Calls
                time.sleep(0.3)

            except Exception as e:
                continue
            
            progress_bar.progress((i + 1) / len(ticker_liste))

        st.session_state.profi_scan_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
        st.rerun()

# --- ANZEIGE DER KACHELN ---
if st.session_state.profi_scan_results:
    res_list = st.session_state.profi_scan_results
    st.subheader(f"🎯 Top-Setups nach Batch-Analyse ({len(res_list)} Treffer)")
    cols = st.columns(4)
    heute_dt = datetime.now()
    
    for idx, res in enumerate(res_list):
        with cols[idx % 4]:
            # Earnings-Check für rote Umrandung
            is_earning_risk = False
            earn_str = res.get('earn', "---")
            if earn_str and earn_str != "---":
                try:
                    parts = earn_str.split('.')
                    earn_date = datetime(heute_dt.year, int(parts[1]), int(parts[0]))
                    if earn_date < heute_dt - timedelta(days=1):
                        earn_date = datetime(heute_dt.year + 1, int(parts[1]), int(parts[0]))
                    if 0 <= (earn_date - heute_dt).days <= 14:
                        is_earning_risk = True
                except: pass

            card_border = "3px solid #ef4444" if is_earning_risk else "1px solid #e5e7eb"
            card_shadow = "0 12px 20px rgba(239, 68, 68, 0.15)" if is_earning_risk else "0 4px 6px rgba(0,0,0,0.05)"
            
            st.markdown(f"""
<div style="background: white; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: {card_shadow}; height: 535px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 1.2em; font-weight: 800;">{res['symbol']} {'⚠️' if is_earning_risk else ''} <span style="color: #f59e0b; font-size: 0.8em;">{res['stars_str']}</span></span>
<span style="font-size: 0.7em; font-weight: 700; color: #3b82f6; background: #3b82f610; padding: 2px 8px; border-radius: 6px;">{res['sent_icon']} {res['status']}</span>
</div>
<hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;">
<div style="margin: 10px 0;">
<div style="font-size: 0.65em; color: #6b7280; font-weight: 600; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 2.1em; font-weight: 900; color: #10b981;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Strike</div><div style="font-size: 0.9em; font-weight: 700;">{res['strike']:.1f}$</div></div>
<div style="border-left: 3px solid #3b82f6; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Puffer</div><div style="font-size: 0.9em; font-weight: 700;">{res['puffer']:.1f}%</div></div>
</div>
<div style="background: {'#fff1f2' if is_earning_risk else '#f0fdf4'}; padding: 8px; border-radius: 8px; border: 1px dashed {'#ef4444' if is_earning_risk else '#10b981'}; margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; font-size: 0.65em; font-weight: bold;">
<span>Stat. Sicherheit (EM):</span><span style="color: {'#ef4444' if is_earning_risk else '#10b981'};">{res['em_safety']:.1f}x</span>
</div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.7em; color: #4b5563; margin-bottom: 10px;">
<span>⏳ <b>{res['tage']}d</b></span>
<span style="background: #f3f4f6; padding: 1px 5px; border-radius: 4px;">RSI: {int(res['rsi'])}</span>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">{'🗓️'} {res['earn']}</span>
</div>
<div style="background: {res['analyst_color']}15; color: {res['analyst_color']}; padding: 8px; border-radius: 8px; font-size: 0.7em; font-weight: bold; text-align: center; border-left: 4px solid {res['analyst_color']};">
{res['analyst_label']}
</div>
</div>
""", unsafe_allow_html=True)
                    
# --- SEKTION 2: DEPOT-MANAGER (STABILISIERTE VERSION) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    st.info("📦 Die Depot-Analyse ist aktuell pausiert, um den Start zu beschleunigen.")
    if st.button("🚀 Depot jetzt analysieren (Inkl. Pivot-Check)", use_container_width=True):
        with st.spinner("Berechne Pivot-Punkte und Signale (gedrosselt)..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "GTM": [100, 17.00], "HIMS": [100, 36.00],
                "HOOD": [100, 120.00], "JKS": [100, 50.00], "NVO": [100, 97.00],
                "RBRK": [100, 70.00], "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    # Sicherheits-Pause für Yahoo
                    time.sleep(1.2) 
                    res = get_stock_data_full(symbol)
                    if res is None or res[0] is None: continue
                    
                    price, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # Sterne & KI (Nutzen intern get_tk)
                    ki_status, _, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
                    
                    # Analysten-Check via info (Session-basiert)
                    info_temp = get_tk(symbol).info
                    analyst_txt_temp, _ = get_analyst_conviction(info_temp)
                    stars_count = 3 if "HYPER" in analyst_txt_temp else 2 if "Stark" in analyst_txt_temp else 1
                    star_display = "⭐" * stars_count

                    # Pivot Logik
                    s2_d = pivots.get('S2') if pivots else None
                    s2_w = pivots.get('W_S2') if pivots else None
                    r2_d = pivots.get('R2') if pivots else None
                    
                    put_action = "⏳ Warten"
                    if rsi < 35 or (s2_d and price <= s2_d * 1.02): put_action = "🟢 JETZT (S2/RSI)"
                    if s2_w and price <= s2_w * 1.01: put_action = "🔥 EXTREM (Weekly S2)"
                    
                    call_action = "⏳ Warten"
                    if rsi > 55 and r2_d and price >= r2_d * 0.98: call_action = "🟢 JETZT (R2/RSI)"

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
                except:
                    continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    col_header, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Daten aktualisieren"):
            st.session_state.depot_data_cache = None
            st.rerun()
    st.table(pd.DataFrame(st.session_state.depot_data_cache))

# --- SEKTION 3: DESIGN-UPGRADE & SICHERHEITS-AMPEL (STABILISIERT) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")
symbol_input = st.text_input("Ticker Symbol", value="MU", help="Gib ein Ticker-Symbol ein").upper()

if symbol_input:
    try:
        with st.spinner(f"Erstelle Dashboard für {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res_full = get_stock_data_full(symbol_input)

            if res_full[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res_full
                analyst_txt, analyst_col = get_analyst_conviction(info)
                heute_dt = datetime.now()

                # --- 1. EARNINGS-PRÜFUNG (14-TAGE-LOGIK) ---
                is_earning_risk = False
                if earn and earn != "---":
                    try:
                        parts = earn.split('.')
                        e_day, e_month = int(parts[0]), int(parts[1])
                        earn_date = datetime(heute_dt.year, e_month, e_day)
                        if earn_date < heute_dt - timedelta(days=1):
                            earn_date = datetime(heute_dt.year + 1, e_month, e_day)
                        
                        tage_bis_earn = (earn_date - heute_dt).days
                        if 0 <= tage_bis_earn <= 14:
                            is_earning_risk = True
                            st.error(f"⚠️ **Earnings-Warnung:** Zahlen am {earn} (in {tage_bis_earn} Tagen)! Hohes Risiko.")
                        else:
                            st.info(f"🗓️ Nächste Earnings: {earn} (noch {tage_bis_earn} Tage)")
                    except:
                        st.info(f"🗓️ Nächste Earnings: {earn}")
                else:
                    st.write("🗓️ Keine Earnings-Daten verfügbar")

                # --- 2. STRATEGIE-SIGNAL (S2 & RSI) ---
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
                
                put_action_scanner = "⏳ Warten (Kein Signal)"
                signal_color = "#64748b" # Neutral Grau

                if s2_w and price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ef4444" 
                elif rsi < 35 or (s2_d and price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60"

                st.markdown(f"""
                    <div style="padding:10px; border-radius:10px; border: 2px solid {signal_color}; text-align:center; margin-bottom: 20px;">
                        <small style="color: #64748b;">Aktuelles Short Put Signal:</small><br>
                        <strong style="font-size:20px; color:{signal_color};">{put_action_scanner}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- 3. AMPEL-LOGIK & STERNE ---
                stars = 0
                if "HYPER" in analyst_txt: stars = 3
                elif "Stark" in analyst_txt: stars = 2
                elif "Neutral" in analyst_txt: stars = 1
                if uptrend and stars > 0: stars += 0.5
                
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif is_earning_risk:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: EARNINGS-GEFAHR"
                elif stars >= 2.5 and uptrend and 30 <= rsi <= 60:
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Sicher)"
                elif "Warnung" in analyst_txt:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ANALYSTEN-WARNUNG"

                # Anzeige Ampel
                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="margin:0; font-size: 1.8em; letter-spacing: 1px;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # --- 4. METRIKEN-BOARD ---
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

                # --- 5. PIVOT ANALYSE ---
                st.markdown("---")
                if pivots_res:
                    st.markdown("#### 🛡️ Technische Absicherung & Ziele (Pivots)")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Weekly S2", f"{pivots_res['W_S2']:.2f} $")
                    pc2.metric("Daily S2", f"{pivots_res['S2']:.2f} $")
                    pc3.metric("Pivot (P)", f"{pivots_res['P']:.2f} $")
                    pc4.metric("Daily R2", f"{pivots_res['R2']:.2f} $")
                    pc5.metric("Weekly R2", f"{pivots_res['W_R2']:.2f} $")
                    st.caption(f"💡 **Tipp:** CC am Weekly R2 ({pivots_res['W_R2']:.2f} $) oder Put am Weekly S2 ({pivots_res['W_S2']:.2f} $).")

                # --- 6. ANALYSTEN BOX ---
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                        <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                        <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                        <hr style="margin: 10px 0;">
                        <span style="color: #555;">📅 Nächste Earnings: <b>{earn if earn else 'n.a.'}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # --- 7. OPTION-CHAIN AUSWAHL (FIX: KEINE LEEREN TABELLEN MEHR) ---
                st.markdown("---")
                st.markdown("### 🎯 Option-Chain Auswahl")
                option_mode = st.radio("Strategie wählen:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                # Zeitfenster 5 bis 60 Tage
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute_dt).days <= 60]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Verfallstag wählen", valid_dates)
                    days_to_expiry = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - heute_dt).days)
                    
                    # Rohdaten holen
                    opt_chain = tk.option_chain(target_date)
                    df_disp = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    
                    if not df_disp.empty:
                        # Schritt A: Relevante Spalten sichern & Mid-Preis berechnen (Fix für 0.00$ Bids)
                        df_disp = df_disp.copy()
                        df_disp['mid'] = (df_disp['bid'] + df_disp['ask']) / 2
                        # Falls Bid 0 ist, nehmen wir das Mid für die Rendite-Berechnung
                        df_disp['calc_price'] = df_disp.apply(lambda x: x['mid'] if x['bid'] <= 0.01 else x['bid'], axis=1)

                        # Schritt B: Filtern nach Seite (Put unter Kurs / Call über Kurs)
                        if "Put" in option_mode:
                            df_disp = df_disp[df_disp['strike'] <= price * 1.02] # Leicht über Kurs starten für ATM
                            df_disp['Puffer %'] = ((price - df_disp['strike']) / price) * 100
                            df_disp = df_disp.sort_values('strike', ascending=False)
                        else:
                            df_disp = df_disp[df_disp['strike'] >= price * 0.98] # Leicht unter Kurs starten für ATM
                            df_disp['Puffer %'] = ((df_disp['strike'] - price) / price) * 100
                            df_disp = df_disp.sort_values('strike', ascending=True)

                        # Schritt C: Liquiditäts-Fallback (Wenn OI > 5 zu leer ist, nimm OI > 0)
                        df_rich = df_disp[df_disp['openInterest'] >= 5].copy()
                        if df_rich.empty:
                            df_rich = df_disp.copy() # Zeige alles, wenn wenig Handel da ist

                        # Schritt D: Rendite p.a. berechnen
                        df_rich['Yield p.a. %'] = (df_rich['calc_price'] / df_rich['strike']) * (365 / days_to_expiry) * 100
                        
                        # Schritt E: Finale Formatierung
                        result_df = df_rich[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(20)

                        def style_rows(row):
                            p = row['Puffer %']
                            if "Put" in option_mode:
                                if p >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                                elif 5 <= p < 10: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                            else: # Call Logik (Puffer hier = Abstand nach oben)
                                if p >= 5: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                            return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                        # ... (Ende deiner Tabellen-Logik)
                        if not result_df.empty:
                            st.dataframe(
                                result_df.style.apply(style_rows, axis=1).format({
                                    'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                                    'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                                }), 
                                use_container_width=True, height=550
                            )
                        else:
                            st.warning("Keine passenden Strikes mit den gewählten Kriterien gefunden.")
                    else:
                        st.error("Keine Optionsdaten für dieses Datum verfügbar.")
                else:
                    st.info("Keine validen Verfallstage im Zeitraum 5-60 Tage gefunden.")

    except Exception as e:
        st.error(f"Fehler bei der Detail-Analyse: {e}")
        st.info("Hinweis: Manche Ticker-Symbole liefern am Wochenende keine Live-Daten.")

# --- FOOTER (Ganz am Ende der Datei) ---
st.divider()
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Daten: Yahoo Finance | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live'}")




