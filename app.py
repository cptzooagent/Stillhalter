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
    
    only_uptrend = st.checkbox("Nur Aufwärtstrend", value=False, key="trend_sid")
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")

    st.markdown("---")
    # WICHTIG: Ticker-Liste muss in die Sidebar, damit du sie im Blick hast
    default_tickers = "MU, LRCX, AMD, NVDA, TSLA, PLTR, COIN, AFRM, ELF, ETSY, GTLB, HIMS, HOOD"
    ticker_input = st.text_area("Ticker-Liste", value=default_tickers)
    ticker_liste = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

# --- 2. MARKT-DATEN FUNKTIONEN ---
@st.cache_data(ttl=1800) # Speichert das Ergebnis für 30 Minuten (1800 Sek)
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

# --- SEKTION: PROFI-SCANNER LOGIK (KEY-ERROR FIX) ---

if 'profi_scan_results' not in st.session_state:
    st.session_state.profi_scan_results = []

if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    with st.spinner("Analysiere Optionen im Fenster 9-25 Tage..."):
        if test_modus:
            ticker_liste_to_scan = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste_to_scan = ticker_liste 

        all_results = []

        def check_single_stock(symbol):
            import pandas as pd
            from datetime import datetime
            
            try:
                tk = yf.Ticker(symbol, session=secure_session)
                fast = tk.fast_info
                cp = fast.last_price
                
                # Standard-Werte für Sicherheit
                days_display = 20
                bid = 0.0
                strike_price = cp * 0.9
                yield_pa = 0.0
                puffer_val = otm_puffer_slider
                rsi_val = 50
                uptrend = False
                
                # --- OPTIONS-SUCHE (9-25 TAGE) ---
                exp_dates = tk.options
                if exp_dates:
                    today = datetime.now().date()
                    best_date = None
                    for d_str in exp_dates:
                        d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
                        diff = (d_obj - today).days
                        if 7 <= diff <= 27:
                            best_date = d_str
                            days_display = diff
                            break
                    
                    if not best_date:
                        best_date = exp_dates[0]
                        days_display = max((datetime.strptime(best_date, '%Y-%m-%d').date() - today).days, 1)

                    opt = tk.option_chain(best_date)
                    puts = opt.puts
                    
                    if not puts.empty:
                        target_strike = cp * (1 - (otm_puffer_slider / 100))
                        idx = (puts['strike'] - target_strike).abs().idxmin()
                        m_put = puts.loc[idx]
                        strike_price = m_put['strike']
                        bid = m_put['bid'] if m_put['bid'] > 0.05 else m_put['lastPrice']
                        bid = max(bid, 0.01)
                        puffer_val = ((cp - strike_price) / cp) * 100
                        yield_pa = (bid / strike_price) * (365 / days_display) * 100

                # --- TECHNIK & FUNDAMENTALS ---
                hist = tk.history(period="250d")
                if not hist.empty and len(hist) >= 200:
                    sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                    sma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    uptrend = cp > sma200
                    
                    delta_close = hist['Close'].diff()
                    gain = (delta_close.where(delta_close > 0, 0)).rolling(window=14).mean()
                    loss = (-delta_close.where(delta_close < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_val = int(100 - (100 / (1 + rs.iloc[-1])))

                inf = tk.info
                rev_growth = inf.get('revenueGrowth', 0) * 100
                mkt_cap = inf.get('marketCap', 0) / 1e9
                earn_ts = inf.get('nextEarningsDate')
                earn_str = datetime.fromtimestamp(earn_ts).strftime("%d.%m.%Y") if earn_ts else "---"

                # Status & EM-Sicherheit (Berechnung für die Kacheln)
                # Wir schätzen die Erwartete Bewegung (EM) konservativ auf 10%, falls keine Vola-Daten da sind
                em_pct = 10.0 
                em_safety = puffer_val / em_pct

                if uptrend and cp >= sma50:
                    t_status, t_icon, t_col = "Trend", "🛡️", "#10b981"
                elif uptrend and cp < sma50:
                    t_status, t_icon, t_col = "Dip", "💎", "#3b82f6"
                else:
                    t_status, t_icon, t_col = "Abwärts", "⚠️", "#ef4444"

                s_val = 5 if rev_growth >= 40 else 4 if rev_growth >= 20 else 3 if rev_growth >= 10 else 2

                # WICHTIG: Hier müssen alle Keys enthalten sein, die die Kacheln abfragen!
                return {
                    'symbol': symbol, 'price': cp, 'y_pa': yield_pa, 'strike': strike_price,
                    'puffer': puffer_val, 'bid': bid, 'delta': 0.15, 'rsi': rsi_val,
                    'earn': earn_str, 'tage': days_display, 'stars_val': s_val, 'stars_str': "⭐" * s_val,
                    'trend_status': t_status, 'trend_icon': t_icon, 'trend_color': t_col,
                    'growth_label': f"Wachstum: {rev_growth:.0f}%", 'mkt_cap': mkt_cap,
                    'em_safety': em_safety, 'em_pct': em_pct # Diese Keys haben gefehlt!
                }
            except Exception:
                return None

        for s in ticker_liste_to_scan:
            res = check_single_stock(s)
            if res:
                if only_uptrend:
                    if res['trend_status'] in ["Trend", "Dip"]:
                        all_results.append(res)
                else:
                    all_results.append(res)
        
        st.session_state.profi_scan_results = sorted(
            all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True
        )

# --- 3. ANZEIGE-SCHLEIFE (HTML) ---
if st.session_state.profi_scan_results:
    res_list = st.session_state.profi_scan_results
    cols = st.columns(4)
    heute_dt = datetime.now()

    for idx, res in enumerate(res_list):
        with cols[idx % 4]:
            # Farben & Risikoparameter
            t_col, t_icon, t_status = res['trend_color'], res['trend_icon'], res['trend_status']
            delta_val = abs(res['delta'])
            delta_col = "#10b981" if delta_val < 0.20 else "#f59e0b" if delta_val < 0.30 else "#ef4444"
            em_col = "#10b981" if res['em_safety'] >= 1.5 else "#f59e0b" if res['em_safety'] >= 1.0 else "#ef4444"
            
            # Earnings-Check für Rahmen
            is_earning_risk = False
            try:
                e_dt = datetime.strptime(res['earn'], "%d.%m.%Y")
                if 0 <= (e_dt - heute_dt).days <= 14: is_earning_risk = True
            except: pass
            card_border = "3px solid #ef4444" if is_earning_risk else "1px solid #e5e7eb"

            html_code = f"""
<div style="background: white; border: {card_border}; border-radius: 16px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif; min-height: 460px; display: flex; flex-direction: column; justify-content: space-between;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="font-size: 1.2em; font-weight: 800; color: #111827;">{res['symbol']}</span>
<span style="font-size: 0.9em;">{res['stars_str']}</span>
</div>
<div style="display: flex; align-items: center; gap: 4px; color: {t_col}; font-weight: 700; font-size: 0.8em; background: {t_col}10; padding: 2px 8px; border-radius: 6px;">
<span>{t_icon}</span><span style="text-transform: uppercase;">{t_status}</span>
</div>
</div>
<div style="margin: 10px 0;">
<div style="font-size: 0.7em; color: #6b7280; font-weight: 600; text-transform: uppercase;">Yield p.a.</div>
<div style="font-size: 2.2em; font-weight: 900; color: #111827;">{res['y_pa']:.1f}%</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
<div style="border-left: 3px solid #8b5cf6; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Strike</div><div style="font-size: 1.0em; font-weight: 700;">{res['strike']:.1f}$</div></div>
<div style="border-left: 3px solid #f59e0b; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Mid</div><div style="font-size: 1.0em; font-weight: 700;">{res['bid']:.2f}$</div></div>
<div style="border-left: 3px solid #3b82f6; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Puffer</div><div style="font-size: 1.0em; font-weight: 700;">{res['puffer']:.1f}%</div></div>
<div style="border-left: 3px solid {delta_col}; padding-left: 8px;"><div style="font-size: 0.6em; color: #6b7280;">Delta</div><div style="font-size: 1.0em; font-weight: 700; color: {delta_col};">{delta_val:.2f}</div></div>
</div>
<div style="background: {em_col}10; padding: 8px 10px; border-radius: 8px; border: 1px dashed {em_col};">
<div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-size: 0.65em; font-weight: bold;">Stat. Erwartung:</span><span style="font-size: 0.8em; font-weight: 800; color: {em_col};">±{res['em_pct']:.1f}%</span></div>
<div style="font-size: 0.6em; color: #6b7280;">Sicherheit: <b>{res['em_safety']:.1f}x EM</b></div>
</div>
</div>
<div>
<hr style="border: 0; border-top: 1px solid #f3f4f6; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72em;">
<span>⏳ <b>{res['tage']}d</b></span>
<span style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px;">RSI: {res['rsi']}</span>
<span style="font-weight: 800; color: {'#ef4444' if is_earning_risk else '#6b7280'};">🗓️ {res['earn']}</span>
</div>
<div style="background: {res['growth_color']}; color: {res['growth_text_color']}; padding: 8px; border-radius: 8px; font-size: 0.65em; font-weight: 800; text-align: center; margin-top: 10px;">{res['growth_label']}</div>
</div>
</div>
"""
            st.markdown(html_code, unsafe_allow_html=True)
                    
# --- SEKTION 2: DEPOT-MANAGER (ROBUST & DIREKT) ---
st.markdown("---")
st.header("🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

if 'depot_data_cache' not in st.session_state:
    st.session_state.depot_data_cache = None

if st.session_state.depot_data_cache is None:
    if st.button("🚀 Depot jetzt analysieren (Direkt-Abruf)", use_container_width=True):
        with st.spinner("Berechne technische Daten für Depotwerte..."):
            my_assets = {
                "LRCX": [100, 210], "MU": [100, 390], "AFRM": [100, 76.00], "ELF": [100, 109.00], "ETSY": [100, 67.00],
                "GTLB": [100, 41.00], "HIMS": [100, 36.00], "HOOD": [100, 120.00], 
                "JKS": [100, 50.00], "NVO": [100, 97.00], "RBRK": [100, 70.00], 
                "SE": [100, 170.00], "TTD": [100, 102.00]
            }
            
            depot_list = []
            for symbol, data in my_assets.items():
                try:
                    tk = yf.Ticker(symbol, session=secure_session)
                    # 1. Kursdaten
                    hist = tk.history(period="250d")
                    if hist.empty: continue
                    
                    price = hist['Close'].iloc[-1]
                    
                    # 2. RSI Berechnung (direkt hier)
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    # 3. Pivot Punkte (Daily)
                    high_d = hist['High'].iloc[-2]
                    low_d = hist['Low'].iloc[-2]
                    close_d = hist['Close'].iloc[-2]
                    pp = (high_d + low_d + close_d) / 3
                    s2_d = pp - (high_d - low_d)
                    r2_d = pp + (high_d - low_d)
                    
                    # 4. SMA 200 Trend
                    sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                    uptrend = price > sma200

                    # Performance & Stars
                    qty, entry = data[0], data[1]
                    perf_pct = ((price - entry) / entry) * 100
                    stars = "⭐" * (2 if uptrend else 1)
                    if rsi_val < 35: stars += "⭐"

                    # Reparatur-Logik
                    put_action = "🟢 JETZT (S2)" if price <= s2_d * 1.02 else "⏳ Warten"
                    call_action = "🟢 JETZT (R2)" if price >= r2_d * 0.98 else "⏳ Warten"

                    depot_list.append({
                        "Ticker": f"{symbol} {stars}",
                        "Einstand": f"{entry:.2f} $",
                        "Aktuell": f"{price:.2f} $",
                        "P/L %": f"{perf_pct:+.1f}%",
                        "RSI": int(rsi_val),
                        "Short Put (Repair)": put_action,
                        "Covered Call": call_action,
                        "S2 Daily": f"{s2_d:.2f} $",
                        "R2 Daily": f"{r2_d:.2f} $",
                        "Trend": "🛡️" if uptrend else "⚠️"
                    })
                except Exception as e:
                    continue
            
            st.session_state.depot_data_cache = depot_list
            st.rerun()

else:
    # Anzeige der Tabelle
    st.table(pd.DataFrame(st.session_state.depot_data_cache))
    if st.button("🔄 Daten zurücksetzen"):
        st.session_state.depot_data_cache = None
        st.rerun()
                    
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


















