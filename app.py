import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

# --- 1. MATHE: DELTA-BERECHNUNG ---
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

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        full_list = list(set(tickers + nasdaq_extra))
        return [t.replace('.', '-') for t in full_list]
    except Exception as e:
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0
            
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        
        # Indikatoren
        rsi_series = calculate_rsi(hist['Close'])
        rsi_val = rsi_series.iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        
        # Bollinger & ATR
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
            
        # Earnings-Check
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
                elif hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                    earn_str = cal['Earnings Date'].iloc[0].strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

# --- UI: SIDEBAR (KOMPLETT-REPARATUR) ---
with st.sidebar:
    st.header("🛡️ Strategie-Einstellungen")
    
    # Basis-Filter
    otm_puffer_slider = st.slider("Gewünschter Puffer (%)", 3, 25, 10, key="puffer_sid")
    min_yield_pa = st.number_input("Mindestrendite p.a. (%)", 0, 100, 15, key="yield_sid")
    
    # Der Aktienpreis-Regler
    min_stock_price, max_stock_price = st.slider(
        "Aktienpreis-Spanne ($)", 
        0, 1000, (20, 500), 
        key="price_sid"
    )

    st.markdown("---")
    st.subheader("Qualitäts-Filter")
    
    # Marktkapitalisierung & Trend
    min_mkt_cap = st.slider("Mindest-Marktkapitalisierung (Mrd. $)", 1, 1000, 50, key="mkt_cap_sid")
    only_uptrend = st.checkbox("Nur Aufwärtstrend (SMA 200)", value=False, key="trend_sid")
    
    # WICHTIG: Die Checkbox für den Simulationsmodus
    test_modus = st.checkbox("🛠️ Simulations-Modus (Test)", value=False, key="sim_checkbox")
    
    st.markdown("---")
    st.info("💡 Profi-Tipp: Für den S&P 500 Scan ab 16:00 Uhr 'Simulations-Modus' deaktivieren.")
    
# --- HILFSFUNKTIONEN FÜR DAS DASHBOARD ---

def get_market_data():
    try:
        # Nasdaq & VIX
        ndq = yf.Ticker("^IXIC")
        vix = yf.Ticker("^VIX")
        btc = yf.Ticker("BTC-USD")
        
        h_ndq = ndq.history(period="60d")
        h_vix = vix.history(period="1d")
        h_btc = btc.history(period="1d")
        
        # Nasdaq Berechnung
        cp_ndq = h_ndq['Close'].iloc[-1]
        sma20_ndq = h_ndq['Close'].rolling(window=20).mean().iloc[-1]
        dist_ndq = ((cp_ndq - sma20_ndq) / sma20_ndq) * 100
        
        # Nasdaq RSI
        delta = h_ndq['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_ndq = 100 - (100 / (1 + rs)).iloc[-1]
        
        # VIX & BTC
        v_val = h_vix['Close'].iloc[-1]
        b_val = h_btc['Close'].iloc[-1]
        
        return cp_ndq, rsi_ndq, dist_ndq, v_val, b_val
    except:
        return 0, 50, 0, 20, 0

def get_crypto_fg():
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/")
        return int(r.json()['data'][0]['value'])
    except:
        return 50

# --- HAUPT-BLOCK: GLOBAL MONITORING ---

st.markdown("## 🌍 Globales Markt-Monitoring")

# Daten abrufen
cp_ndq, rsi_ndq, dist_ndq, vix_val, btc_val = get_market_data()
crypto_fg = get_crypto_fg()
# Hinweis: fg_val (Stock) müsstest du aus deiner vorhandenen Funktion nehmen
stock_fg = 50 # Platzhalter, falls deine Funktion anders heißt

# Master-Banner Logik
if dist_ndq < -2 or vix_val > 25:
    m_color, m_text = "#e74c3c", "🚨 MARKT-ALARM: Nasdaq-Schwäche / Hohe Volatilität"
    m_advice = "Defensiv agieren. Fokus auf Call-Verkäufe zur Depot-Absicherung."
elif rsi_ndq > 72 or stock_fg > 80:
    m_color, m_text = "#f39c12", "⚠️ ÜBERHITZT: Korrekturgefahr (Gier/RSI hoch)"
    m_advice = "Keine neuen Puts mit engem Puffer. Gewinne sichern."
else:
    m_color, m_text = "#27ae60", "✅ TRENDSTARK: Marktumfeld ist konstruktiv"
    m_advice = "Puts auf starke Aktien bei Rücksetzern möglich."

st.markdown(f"""
    <div style="background-color: {m_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; font-size: 1.4em;">{m_text}</h3>
        <p style="margin:0; opacity: 0.9;">{m_advice}</p>
    </div>
""", unsafe_allow_html=True)

# Das 2x3 Raster
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1:
    st.metric("Nasdaq 100", f"{cp_ndq:,.0f}", f"{dist_ndq:.1f}% vs SMA20")
with r1c2:
    st.metric("Bitcoin", f"{btc_val:,.0f} $")
with r1c3:
    st.metric("VIX (Angst)", f"{vix_val:.2f}", delta="HOCH" if vix_val > 22 else "Normal", delta_color="inverse")

with r2c1:
    st.metric("Fear & Greed (Stock)", f"{stock_fg}")
with r2c2:
    st.metric("Fear & Greed (Crypto)", f"{crypto_fg}")
with r2c3:
    st.metric("Nasdaq RSI (14)", f"{int(rsi_ndq)}", delta="HEISS" if rsi_ndq > 70 else None, delta_color="inverse")

st.markdown("---")


# --- NEUE ANALYSTEN-LOGIK (VOR DEM SCAN DEFINIEREN) ---
def get_analyst_conviction(info):
    try:
        current = info.get('current_price', info.get('currentPrice', 1))
        target = info.get('targetMedianPrice', 0)
        upside = ((target / current) - 1) * 100 if target > 0 else 0
        rev_growth = info.get('revenueGrowth', 0) * 100
        
        # 1. LILA: 🚀 HYPER-GROWTH (APP, CRDO, NVDA, ALAB)
        if rev_growth > 40:
            return f"🚀 HYPER-GROWTH (+{rev_growth:.0f}% Wachst.)", "#9b59b6"
        # 2. GRÜN: ✅ STARK (AVGO, CRWD)
        elif upside > 15 and rev_growth > 5:
            return f"✅ Stark (Ziel: +{upside:.0f}%, Wachst.: {rev_growth:.1f}%)", "#27ae60"
        # 3. BLAU: 💎 QUALITY-DIP (NET, MRVL)
        elif upside > 25:
            return f"💎 Quality-Dip (Ziel: +{upside:.0f}%)", "#2980b9"
        # 4. ORANGE: ⚠️ WARNUNG (GTM, stagnierende Werte)
        elif upside < 0 or rev_growth < -2:
            return f"⚠️ Warnung (Ziel: {upside:.1f}%, Wachst.: {rev_growth:.1f}%)", "#e67e22"
        return f"⚖️ Neutral (Ziel: {upside:.0f}%)", "#7f8c8d"
    except:
        return "🔍 Check nötig", "#7f8c8d"



# --- SEKTION: PROFI-SCANNER (RSI & STERNE EDITION) ---
if st.button("🚀 Profi-Scan starten", key="kombi_scan_pro"):
    puffer_limit = otm_puffer_slider / 100 
    mkt_cap_limit = min_mkt_cap * 1_000_000_000
    
    with st.spinner("Lade Ticker-Liste..."):
        if test_modus:
            # HIER SIND ALLE DEINE SYMBOLE GESICHERT:
            ticker_liste = ["APP", "AVGO", "NET", "CRWD", "MRVL", "NVDA", "CRDO", "HOOD", "SE", "ALAB", "TSLA", "PLTR", "COIN", "MSTR", "TER", "DELL", "DDOG", "MU", "LRCX", "RTX", "UBER"]
        else:
            ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    for i, symbol in enumerate(ticker_liste):
        progress_bar.progress((i + 1) / len(ticker_liste))
        if i % 3 == 0: 
            status_text.text(f"Analysiere {i}/{len(ticker_liste)}: {symbol}...")
        
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            curr_price = info.get('currentPrice', 0)
            m_cap = info.get('marketCap', 0)
            
            if m_cap < mkt_cap_limit: continue
            if not (min_stock_price <= curr_price <= max_stock_price): continue
            
            res = get_stock_data_full(symbol)
            if res[0] is None: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if only_uptrend and not uptrend: continue

            # Laufzeit & Earnings-Schutz
            heute = datetime.now()
            max_days_allowed = 24
            if earn and "." in earn:
                try:
                    tag, monat = earn.split(".")[:2]
                    er_datum = datetime(heute.year, int(monat), int(tag))
                    if er_datum < heute: er_datum = datetime(heute.year + 1, int(monat), int(tag))
                    max_days_allowed = min(24, (er_datum - heute).days - 2)
                except: pass

            valid_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= max_days_allowed]
            if not valid_dates: continue
            
            target_date = valid_dates[0]
            chain = tk.option_chain(target_date).puts
            target_strike = price * (1 - puffer_limit)
            opts = chain[chain['strike'] <= target_strike].sort_values('strike', ascending=False)
            
            if not opts.empty:
                o = opts.iloc[0]
                bid_val = o['bid'] if o['bid'] > 0 else o['lastPrice']
                days = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                y_pa = (bid_val / o['strike']) * (365 / max(1, days)) * 100
                
                if y_pa >= min_yield_pa:
                    analyst_txt, analyst_col = get_analyst_conviction(info)
                    
                    # Sterne Logik (Priorität)
                    stars = 0
                    if "HYPER" in analyst_txt: stars = 3
                    elif "Stark" in analyst_txt: stars = 2
                    elif "Neutral" in analyst_txt: stars = 1
                    if uptrend and stars > 0: stars += 0.5 

                    all_results.append({
                        'symbol': symbol, 'price': price, 'y_pa': y_pa, 
                        'strike': o['strike'], 'puffer': ((price - o['strike']) / price) * 100,
                        'bid': bid_val, 'rsi': rsi, 'earn': earn if earn else "n.a.", 
                        'tage': days, 'status': "🛡️ Trend" if uptrend else "💎 Dip",
                        'stars_val': stars, 'stars_str': "⭐" * int(stars),
                        'analyst_txt': analyst_txt, 'analyst_col': analyst_col,
                        'mkt_cap': m_cap / 1_000_000_000
                    })
        except: continue

    status_text.empty()
    progress_bar.empty()

    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        # Sortierung: Qualität (Sterne) -> Rendite
        all_results = sorted(all_results, key=lambda x: (x['stars_val'], x['y_pa']), reverse=True)

        st.markdown(f"### 🎯 Top-Setups ({len(all_results)} Treffer)")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                s_color = "#27ae60" if "🛡️" in res['status'] else "#2980b9"
                border_color = res['analyst_col'] if res['stars_val'] >= 2 else "#e0e0e0"
                rsi_col = "#e74c3c" if res['rsi'] > 70 else ("#27ae60" if res['rsi'] < 30 else "#7f8c8d")
                
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
                    
# --- SEKTION 4: DEPOT-MANAGER MIT EINGABE-FORMULAR ---
st.markdown("### 🛠️ Depot-Manager: Bestandsverwaltung & Reparatur")

# 1. Speicher für das Depot initialisieren (falls noch nicht vorhanden)
if 'my_portfolio_data' not in st.session_state:
    st.session_state.my_portfolio_data = []

# 2. EINGABE-BEREICH (Formular)
with st.expander("➕ Neuen Depot-Wert hinzufügen"):
    with st.form("add_stock_form"):
        col_t, col_p, col_s = st.columns(3)
        new_ticker = col_t.text_input("Ticker Symbol", placeholder="z.B. HIMS").upper()
        new_price = col_p.number_input("Einstandskurs ($)", min_value=0.0, step=0.1)
        new_shares = col_s.number_input("Stückzahl", min_value=0, step=1)
        
        if st.form_submit_button("Aktie zum Depot hinzufügen"):
            if new_ticker:
                # Neuen Wert zur Liste hinzufügen
                st.session_state.my_portfolio_data.append({
                    "symbol": new_ticker, 
                    "buy_price": new_price, 
                    "shares": new_shares
                })
                st.success(f"{new_ticker} wurde hinzugefügt!")
            else:
                st.error("Bitte einen Ticker eingeben.")

# 3. ANZEIGE & ANALYSE DER DEPOTWERTE
if not st.session_state.my_portfolio_data:
    st.info("Dein Depot ist noch leer. Füge oben deine ersten Werte hinzu.")
else:
    # Button zum Leeren des Depots (optional)
    if st.button("Depot-Liste leeren"):
        st.session_state.my_portfolio_data = []
        st.rerun()

    # Analyse für jeden eingetragenen Wert
    for stock in st.session_state.my_portfolio_data:
        symbol = stock["symbol"]
        buy_price = stock["buy_price"]
        
        with st.expander(f"📊 Position: {symbol} (Einstieg: {buy_price:.2f} $)", expanded=True):
            res = get_stock_data_full(symbol)
            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr = res
                
                # Performance & Ampel-Logik
                profit_loss_pct = ((price - buy_price) / buy_price) * 100
                
                if rsi < 25:
                    status_txt, status_col = "🚨 PANIK: Keine Calls verkaufen!", "#c0392b"
                elif rsi > 55:
                    status_txt, status_col = "✅ ERHOLUNG: Call-Verkauf prüfen", "#27ae60"
                else:
                    status_txt, status_col = "🟡 GEDULD: RSI neutral", "#f1c40f"

                # Visuelle Aufbereitung
                st.markdown(f"""
                    <div style="background:{status_col}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold;">
                        {status_txt}
                    </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Kurs", f"{price:.2f} $")
                c2.metric("P/L %", f"{profit_loss_pct:.1f} %", delta=f"{price-buy_price:.2f} $")
                c3.metric("RSI", int(rsi))
                
                # Strategie-Check für Calls
                if st.checkbox(f"Reparatur-Calls für {symbol} anzeigen"):
                    st.write(f"Suche Calls mit Strike >= **{buy_price:.2f} $** (Break-Even Schutz)")
                    # ... [Hier folgt dein bestehender Code für die Call-Tabelle] ...
                    
# --- NEUE HILFSFUNKTION FÜR PIVOT-PUNKTE (Ganz oben bei den Funktionen einfügen) ---
def calculate_pivots(symbol):
    try:
        tk = yf.Ticker(symbol)
        # Wir benötigen 2 Tage, um den abgeschlossenen Vortag zu analysieren
        hist = tk.history(period="2d")
        if len(hist) < 2: return None
        
        # Daten des letzten abgeschlossenen Handelstages
        last_day = hist.iloc[-2] 
        h, l, c = last_day['High'], last_day['Low'], last_day['Close']
        
        pp = (h + l + c) / 3
        s1 = (2 * pp) - h
        s2 = pp - (h - l)
        return {"PP": pp, "S1": s1, "S2": s2}
    except:
        return None

# --- SEKTION 3: PROFI-ANALYSE & TRADING-COCKPIT (STABILE VERSION) ---
st.markdown("### 🔍 Profi-Analyse & Trading-Cockpit")

col_input, col_switch = st.columns([2, 2])
with col_input:
    symbol_input = st.text_input("Ticker Symbol", value="MU").upper()
with col_switch:
    strategy_mode = st.radio("Strategie-Modus", ["🛡️ Short Put", "🛠️ Short Call"], horizontal=True)

if symbol_input:
    try:
        with st.spinner(f"Analysiere {symbol_input}..."):
            tk = yf.Ticker(symbol_input)
            info = tk.info
            res = get_stock_data_full(symbol_input)
            pivots = calculate_pivots(symbol_input)
            
            if res[0] is not None:
                price, dates, earn, rsi, uptrend, near_lower, atr = res
                analyst_txt, analyst_col = get_analyst_conviction(info)
                
                # Markt-Penalty Logik (basierend auf Nasdaq-Stand)
                market_penalty = False
                if 'dist_ndq' in locals() and dist_ndq < -1.5:
                    market_penalty = True

                # Ampel-Logik
                ampel_color, ampel_text = "#f1c40f", "NEUTRAL / ABWARTEN"
                if rsi < 25: ampel_color, ampel_text = "#e74c3c", "🚨 PANIK-ABVERKAUF"
                elif market_penalty: ampel_color, ampel_text = "#e74c3c", "🚨 NASDAQ SCHWÄCHE"
                elif strategy_mode == "🛡️ Short Put" and rsi > 70: ampel_color, ampel_text = "#e74c3c", "🚨 ÜBERHITZT"
                else: ampel_color, ampel_text = "#27ae60", "✅ SETUP BEREIT"

                st.markdown(f"""
                    <div style="background-color: {ampel_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin:0; font-size: 1.8em;">● {ampel_text}</h2>
                    </div>
                """, unsafe_allow_html=True)

                # --- OPTION-CHAIN LOGIK (FIXED) ---
                st.markdown(f"### 🎯 {strategy_mode} Auswahl")
                heute = datetime.now()
                valid_dates = [d for d in dates if 5 <= (datetime.strptime(d, '%Y-%m-%d') - heute).days <= 45]
                
                if valid_dates:
                    target_date = st.selectbox("📅 Verfallstag", valid_dates)
                    days_to_expiry = (datetime.strptime(target_date, '%Y-%m-%d') - heute).days
                    
                    try:
                        # Holen der Daten je nach Modus
                        if strategy_mode == "🛡️ Short Put":
                            chain = tk.option_chain(target_date).puts
                        else:
                            chain = tk.option_chain(target_date).calls
                        
                        if not chain.empty and 'bid' in chain.columns:
                            # 1. Spalten berechnen
                            chain['strike'] = chain['strike'].astype(float)
                            chain['Yield p.a. %'] = (chain['bid'] / price) * (365 / max(1, days_to_expiry)) * 100
                            
                            if strategy_mode == "🛡️ Short Put":
                                chain['Puffer %'] = ((price - chain['strike']) / price) * 100
                                df_disp = chain[chain['strike'] < price].copy()
                            else:
                                chain['Puffer %'] = ((chain['strike'] - price) / price) * 100
                                df_disp = chain[chain['strike'] > price].copy()

                            # 2. Sortierung
                            df_disp = df_disp.sort_values('strike', ascending=(strategy_mode == "🛠️ Short Call"))
                            
                            # 3. Anzeige
                            st.dataframe(df_disp[['strike', 'bid', 'ask', 'Puffer %', 'Yield p.a. %']].format({
                                'strike': '{:.2f} $', 'bid': '{:.2f} $', 'ask': '{:.2f} $',
                                'Puffer %': '{:.1f} %', 'Yield p.a. %': '{:.1f} %'
                            }), use_container_width=True)
                        else:
                            st.warning("Keine Live-Daten verfügbar (Börse evtl. geschlossen).")
                            
                    except Exception as opt_e:
                        st.error(f"Fehler bei Optionsdaten: {opt_e}")
                else:
                    st.info("Keine passenden Verfallstage gefunden.")

    except Exception as e:
        st.error(f"Allgemeiner Fehler: {e}")
