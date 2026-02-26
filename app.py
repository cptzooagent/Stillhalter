import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import concurrent.futures
from curl_cffi.requests import Session as FastSession

# MUSS der erste Streamlit-Befehl im Skript sein!
st.set_page_config(
    page_title="Profi-Trading Cockpit",
    page_icon="🚀",
    layout="wide", # Das hier löst das Stauchungs-Problem
    initial_sidebar_state="expanded"
)

# --- 1. INFRASTRUKTUR: SICHERE SESSION GEGEN 401 FEHLER ---
class CurlCffiSession(FastSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def request(self, method, url, *args, **kwargs):
        # Fingerprint-Impersonation um Yahoo-Sperren zu umgehen
        kwargs.setdefault("impersonate", "chrome120")
        return super().request(method, url, *args, **kwargs)

# Globale Session für alle yfinance-Aufrufe
secure_session = CurlCffiSession()

with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    
    # Puffer als Slider für schnelles Adjustieren
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 0, 100, 15, key="puffer_sid")
    p_puffer = otm_puffer_slider / 100
    
    # Rendite als Number Input für Präzision (wie in deiner alten Sidebar)
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 12, key="yield_sid")
    p_min_yield = min_yield_pa

    # Preis-Spanne 0-1000
    min_stock_price, max_stock_price = st.slider("Aktienpreis-Spanne ($)", 0, 1000, (60, 500), key="price_sid")
    
    st.markdown("---")
    st.subheader("Qualitäts-Filter")
    
    # Market Cap 0-1000 (Mrd $)
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 0, 1000, 20, key="mkt_cap_sid")
    p_min_cap = min_mkt_cap * 1e9
    
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")

    st.markdown("---")
    # WICHTIG: Ticker-Liste muss in die Sidebar, damit du sie im Blick hast
    default_tickers = "MU, LRCX, AMD, NVDA, TSLA, PLTR, COIN, AFRM, ELF, ETSY, GTLB, HIMS, HOOD"
    ticker_input = st.text_area("Ticker-Liste", value=default_tickers)
    ticker_liste = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

# --- 2. MARKT-DATEN FUNKTIONEN ---

def get_market_data():
    try:
        # Ticker mit sicherer Session initialisieren
        tickers = {
            "ndq": yf.Ticker("^NDX", session=secure_session),
            "vix": yf.Ticker("^VIX", session=secure_session),
            "btc": yf.Ticker("BTC-USD", session=secure_session)
        }
        
        # Historie für RSI/SMA (Nasdaq)
        h_ndq = tickers["ndq"].history(period="1mo")
        if h_ndq.empty: return 0, 50, 0, 20, 0
        
        # Schnelle Daten via fast_info (Verhindert Timeouts)
        cp_ndq = tickers["ndq"].fast_info.last_price
        v_val = tickers["vix"].fast_info.last_price
        b_val = tickers["btc"].fast_info.last_price
        
        # Technische Berechnungen
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        
        delta = h_ndq['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_ndq = 100 - (100 / (1 + rs)).iloc[-1]
        
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except Exception as e:
        st.error(f"Fehler beim Abruf der Marktdaten: {e}")
        return 0, 50, 0, 20, 0

def get_crypto_fg():
    """Holt den Crypto Fear & Greed Index."""
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/").json()
        return int(r['data'][0]['value'])
    except:
        return 50

def get_stock_fg(vix):
    """Schätzt Stock Fear & Greed basierend auf VIX (Inverse Korrelation)."""
    if vix < 15: return 80  # Extreme Greed
    if vix > 30: return 20  # Extreme Fear
    # Lineare Skalierung zwischen VIX 15 (80) und 30 (20)
    val = 80 - ((vix - 15) * 4)
    return int(max(0, min(100, val)))

# --- 3. UI GLOBAL DASHBOARD ---

st.markdown("## 🌍 Globales Markt-Monitoring")

# Daten abrufen
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
stock_fg = get_stock_fg(vix_val)

# Logik für die Markt-Ampel
if dist_ndq < -2.5 or vix_val > 24:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Instabiles Umfeld"
    m_advice = "Hohes Risiko. Cash-Quote erhöhen oder Covered Calls zur Absicherung rollen."
elif rsi_ndq > 70 or stock_fg > 75:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Rückschlaggefahr"
    m_advice = "Gier nimmt zu. Neue Puts nur mit >20% Puffer und kleiner Positionsgröße."
else:
    m_color, m_text = "#27ae60", "✅ KONSTRUKTIV: Trend intakt"
    m_advice = "Gutes Umfeld für Cash-Secured Puts auf Qualitätsaktien (SMA200-Check)."

# Anzeige der Ampel-Box
st.markdown(f"""
    <div style="background-color: {m_color}; color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px; border-left: 8px solid rgba(0,0,0,0.2);">
        <h3 style="margin:0; font-size: 1.5em; font-weight: bold;">{m_text}</h3>
        <p style="margin:5px 0 0 0; font-size: 1.1em; opacity: 0.9;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

# Grid Layout für Metriken
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1: st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2: st.metric("Bitcoin", f"{btc_val:,.0f} $", delta=f"{crypto_fg}% F&G", delta_color="off")
with r1c3: st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")

with r2c1: 
    fg_label = "Fear" if stock_fg < 40 else "Greed" if stock_fg > 60 else "Neutral"
    st.metric(f"Stock F&G ({fg_label})", f"{stock_fg}")
with r2c2: st.metric("Krypto Sentiment", f"{crypto_fg}%")
with r2c3: st.metric("Nasdaq RSI (14)", f"{int(rsi_ndq)}", delta="HEISS" if rsi_ndq > 70 else "OK", delta_color="inverse")

st.markdown("---")

# --- SEKTION 1: PROFI-SCANNER

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    p_puffer = otm_puffer_slider / 100 
    p_min_yield = min_yield_pa
    p_min_cap = min_mkt_cap * 1_000_000_000
    heute = datetime.now()
    
    with st.spinner("Markt-Scanner analysiert Ticker..."):
        ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"] if test_modus else get_combined_watchlist()
        status_text = st.empty()
        progress_bar = st.progress(0)
        all_results = []

        def check_single_stock(symbol):
            try:
                # Kein künstliches Sleep mehr nötig bei fast_info!
                tk = yf.Ticker(symbol, session=secure_session)
                f_info = tk.fast_info
        
                price = f_info.last_price
                m_cap = f_info.market_cap
        
                # Filter auf Basis der schnellen Daten
                if m_cap < p_min_cap or not (min_stock_price <= price <= max_stock_price): 
                    return None
            
                res = get_stock_data_full(symbol)
                if res is None or res[0] is None: return None
                _, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
        
                if only_uptrend and not uptrend: return None
        
                # Options-Daten ziehen (bleibt bei yf.Ticker, da nicht in fast_info)
                valid_dates = [d for d in dates if 10 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 30]
                if not valid_dates: return None
                target_date = valid_dates[0]
                chain = tk.option_chain(target_date).puts
                target_strike = price * (1 - p_puffer)
                opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
                if opts.empty: return None
                o = opts.iloc[0]
        
                # Expected Move & Delta
                days_to_exp = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                iv = o.get('impliedVolatility', 0.4)
                exp_move_pct = (price * iv * np.sqrt(days_to_exp / 365) / price) * 100
                current_puffer = ((price - o['strike']) / price) * 100
                em_safety = current_puffer / exp_move_pct if exp_move_pct > 0 else 0
        
                # Analysten-Daten (Selektiver Abruf nur bei Treffern!)
                try:
                    full_info = tk.info # Nur hier wird tk.info kurz genutzt
                    analyst_txt, analyst_col = get_analyst_conviction(full_info)
                except:
                    analyst_txt, analyst_col = "🔍 Check nötig", "#7f8c8d"

                y_pa = ((o['bid'] + o['ask'])/2 / o['strike']) * (365 / max(1, days_to_exp)) * 100
        
                if y_pa >= p_min_yield:
                    return {
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': o['strike'], 
                        'puffer': current_puffer, 'bid': (o['bid'] + o['ask'])/2, 'rsi': rsi, 'earn': earn, 
                        'tage': days_to_exp, 'status': "🛡️ Trend" if uptrend else "💎 Dip", 
                        'delta': calculate_bsm_delta(price, o['strike'], days_to_exp/365, iv),
                        'stars_val': 2.0, 'stars_str': "⭐", 'analyst_label': analyst_txt, 
                        'analyst_color': analyst_col, 'mkt_cap': m_cap / 1e9,
                        'em_pct': exp_move_pct, 'em_safety': em_safety
                    }
            except: return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(check_single_stock, s): s for s in ticker_liste}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_data = future.result()
                if res_data: all_results.append(res_data)
                progress_bar.progress((i + 1) / len(ticker_liste))
                if i % 5 == 0: status_text.text(f"Checke {i}/{len(ticker_liste)} Ticker...")
        status_text.empty(); progress_bar.empty()
        if all_results:
            st.session_state.profi_scan_results = sorted(all_results, key=lambda x: (float(x.get('stars_val', 0)), float(x.get('y_pa', 0))), reverse=True)
            st.success(f"Scan abgeschlossen: {len(all_results)} Treffer gefunden!")
        else:
            st.session_state.profi_scan_results = []
            st.warning("Keine Treffer gefunden.")

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
            delta_val = abs(res.get('delta', 0))
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            
            # --- NEU: EM FARBE ---
            em_safety = res.get('em_safety', 1.0)
            em_col = "#10b981" if em_safety >= 1.5 else "#f59e0b" if em_safety >= 1.0 else "#ef4444"
            
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
<span style="font-size: 0.75em; font-weight: 800; color: {em_col};">±{res['em_pct']:.1f}%</span>
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
                    tk = yf.Ticker(symbol, session=secure_session)
                    f_info = tk.fast_info
                    price = f_info.last_price
    
                    res = get_stock_data_full(symbol)
                    if res is None or res[0] is None: continue
    
                    price_full, dates, earn, rsi, uptrend, near_lower, atr, pivots = res
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100

                    # KI & Sentiment (fast_info hat keine News, also bleibt get_openclaw_analysis)
                    ki_status, _, _ = get_openclaw_analysis(symbol)
                    ki_icon = "🟢" if ki_status == "Bullish" else "🔴" if ki_status == "Bearish" else "🟡"
    
                    # Sterne-Ersatz durch Trend/RSI Logik (um tk.info zu meiden)
                    stars_count = 1
                    if uptrend: stars_count += 1
                    if rsi < 40: stars_count += 1
                    star_display = "⭐" * stars_count

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
            # SCHNELLE DATENABFRAGE (fast_info)
            f_info = tk.fast_info
            current_price = f_info.last_price
            
            # TECHNISCHE DATEN (RSI, Pivots, etc.)
            res = get_stock_data_full(symbol_input)

            if res[0] is not None:
                price_res, dates, earn, rsi, uptrend, near_lower, atr, pivots_res = res
                
                # --- EARNINGS-LOGIK ---
                if earn and earn != "---":
                    # Warnung für das aktuelle Quartal (Beispiel 2026)
                    if "Feb" in earn or "Mar" in earn:
                        st.error(f"⚠️ **Earnings-Warnung:** Nächste Zahlen am {earn}. Vorsicht bei neuen Trades!")
                    else:
                        st.info(f"🗓️ Nächste Earnings: {earn}")
                else:
                    st.write("🗓️ Keine Earnings-Daten verfügbar")

                # --- STRATEGIE-SIGNAL (Basis: fast_info Preis) ---
                s2_d = pivots_res.get('S2') if pivots_res else None
                s2_w = pivots_res.get('W_S2') if pivots_res else None
        
                put_action_scanner = "⏳ Warten (Kein Signal)"
                signal_color = "white"

                if s2_w and current_price <= s2_w * 1.01:
                    put_action_scanner = "🔥 EXTREM (Weekly S2)"
                    signal_color = "#ff4b4b" 
                elif rsi < 35 or (s2_d and current_price <= s2_d * 1.02):
                    put_action_scanner = "🟢 JETZT (S2/RSI)"
                    signal_color = "#27ae60"

                st.markdown(f"""
                    <div style="padding:10px; border-radius:10px; border: 2px solid {signal_color}; text-align:center;">
                        <small>Aktuelles Short Put Signal:</small><br>
                        <strong style="font-size:20px; color:{signal_color};">{put_action_scanner}</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- AMPEL-LOGIK (Panik-Schutz) ---
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                
                if rsi < 25:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: PANIK-ABVERKAUF (RSI < 25)"
                elif rsi > 75:
                    ampel_color, ampel_text = "#e74c3c", "STOPP: ÜBERHITZT (RSI > 75)"
                elif uptrend and 30 <= rsi <= 60:
                    ampel_color, ampel_text = "#27ae60", "TOP SETUP (Technisch Stark)"
                else:
                    ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"

                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="margin:0; font-size: 1.8em; letter-spacing: 1px;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # --- METRIKEN-BOARD ---
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Kurs (Fast)", f"{current_price:.2f} $")
                with col2: st.metric("RSI (14)", f"{int(rsi)}", delta="LOW" if rsi < 30 else None, delta_color="inverse")
                with col3: st.metric("Phase", f"{'🛡️ Trend' if uptrend else '💎 Dip'}")
                with col4: st.metric("Volumen (Avg)", f"{f_info.average_volume_10day / 1e6:.1f}M")

                # --- PIVOT ANALYSE ---
                st.markdown("---")
                if pivots_res:
                    st.markdown("#### 🛡️ Technische Absicherung & Ziele (Pivots)")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Weekly S2", f"{pivots_res['W_S2']:.2f} $")
                    pc2.metric("Daily S2", f"{pivots_res['S2']:.2f} $")
                    pc3.metric("Pivot (P)", f"{pivots_res['P']:.2f} $")
                    pc4.metric("Daily R2", f"{pivots_res['R2']:.2f} $")
                    pc5.metric("Weekly R2", f"{pivots_res['W_R2']:.2f} $")

                # --- FUNDAMENTALE BOX (Einziger tk.info Aufruf) ---
                try:
                    info = tk.info
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {analyst_col}; margin-top: 10px;">
                            <h4 style="margin-top:0; color: #31333F;">💡 Fundamentale Analyse</h4>
                            <p style="font-size: 1.1em; font-weight: bold; color: {analyst_col};">{analyst_txt}</p>
                            <hr style="margin: 10px 0;">
                            <span style="color: #555;">Marktkapitalisierung: <b>{f_info.market_cap / 1e9:.1f} Mrd. $</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                except:
                    st.warning("Analysten-Daten aktuell nicht verfügbar.")

                # --- OPTION-CHAIN AUSWAHL ---
                st.markdown("---")
                st.markdown("### 🎯 Option-Chain Auswahl")
                option_mode = st.radio("Strategie wählen:", ["Put (Cash Secured)", "Call (Covered)"], horizontal=True)
                
                heute = datetime.now()
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 35]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Wähle deinen Verfallstag", valid_dates)
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days

                    # KI-Sentiment Check
                    ki_status, ki_text, ki_score = get_openclaw_analysis(symbol_input)
                    st.info(ki_text)

                    opt_chain = tk.option_chain(target_date)
                    chain = opt_chain.puts if "Put" in option_mode else opt_chain.calls
                    df_disp = chain[chain['openInterest'] > 20].copy() # Liqui-Filter
                    
                    if "Put" in option_mode:
                        df_disp = df_disp[df_disp['strike'] < current_price].copy()
                        df_disp['Puffer %'] = ((current_price - df_disp['strike']) / current_price) * 100
                        sort_order = False
                    else:
                        df_disp = df_disp[df_disp['strike'] > current_price].copy()
                        df_disp['Puffer %'] = ((df_disp['strike'] - current_price) / current_price) * 100
                        sort_order = True

                    df_disp['Yield p.a. %'] = (df_disp['bid'] / df_disp['strike']) * (365 / max(1, days_to_expiry)) * 100
                    df_disp = df_disp.sort_values('strike', ascending=sort_order)

                    # Styling & Anzeige
                    def style_rows(row):
                        p = row['Puffer %']
                        if p >= 10: return ['background-color: rgba(39, 174, 96, 0.1)'] * len(row)
                        elif 5 <= p < 10: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)

                    st.dataframe(
                        df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].head(12).style.apply(style_rows, axis=1).format({
                            'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                            'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                        }), use_container_width=True
                    )

    except Exception as e:
        st.error(f"Fehler bei der Detail-Analyse: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')} | Modus: {'🛠️ Simulation' if test_modus else '🚀 Live-Scan'}")







