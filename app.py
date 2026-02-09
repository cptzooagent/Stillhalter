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
    # Sicherheits-Check für Volatilität (IV)
    # Yahoo liefert oft 0.5 (50%) oder 50.0 (50%). Wir normalisieren das.
    if sigma is None or sigma == 0:
        sigma = 0.4
    if sigma > 5: # Wenn IV als 40.0 statt 0.4 kommt
        sigma = sigma / 100
        
    # Sicherheits-Check für Zeit (T)
    if T <= 0:
        T = 1/365 # Mindestens 1 Tag Restlaufzeit für die Berechnung

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    except:
        return 0.0

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

# --- UI: SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
otm_puffer_slider = st.sidebar.slider("Gewünschter Puffer (%)", 3, 25, 10, help="Abstand vom Strike zum Kurs")
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", 0, 100, 15)
min_stock_price, max_stock_price = st.sidebar.slider("Aktienpreis-Spanne ($)", 0, 1000, (20, 500))

st.sidebar.markdown("---")
only_uptrend = st.sidebar.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)
st.sidebar.info("Tipp: Deaktiviere den Aufwärtstrend für mehr Treffer am Wochenende.")

# --- DAS ULTIMATIVE MARKT-DASHBOARD (4 SPALTEN MIT DELTAS) ---
st.markdown("## 📊 Globales Marktwetter")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# 1. Stock Fear & Greed (Proxy via SPY RSI)
try:
    spy_hist = yf.Ticker("SPY").history(period="30d")
    spy_rsi_series = calculate_rsi(spy_hist['Close'])
    rsi_val = spy_rsi_series.iloc[-1]
    rsi_prev = spy_rsi_series.iloc[-2]
    rsi_delta = rsi_val - rsi_prev
    fng_text = "Neutral" if 40 < rsi_val < 60 else "Greed" if rsi_val >= 60 else "Fear"
    m_col1.metric("Stock Sentiment (RSI)", f"{rsi_val:.0f}/100", f"{rsi_delta:+.1f} ({fng_text})")
except:
    m_col1.error("Stock F&G n.a.")

# 2. Crypto Fear & Greed (API)
try:
    import requests
    fg_data = requests.get("https://api.alternative.me/fng/").json()
    fg_crypto = int(fg_data['data'][0]['value'])
    fg_c_text = fg_data['data'][0]['value_classification']
    # Da die API kein Delta liefert, zeigen wir den Text als Delta-Ersatz
    m_col2.metric("Crypto Fear & Greed", f"{fg_crypto}/100", fg_c_text)
except:
    m_col2.error("Crypto F&G n.a.")

# 3. VIX (Angst-Index) mit Änderung
try:
    vix_data = yf.Ticker("^VIX").history(period="2d")
    vix_val = vix_data['Close'].iloc[-1]
    vix_prev = vix_data['Close'].iloc[-2]
    v_delta = vix_val - vix_prev
    # delta_color="inverse": VIX steigt = Rot, VIX sinkt = Grün
    m_col3.metric("VIX (Angst-Index)", f"{vix_val:.2f}", f"{v_delta:+.2f}", delta_color="inverse")
except:
    m_col3.error("VIX n.a.")

# 4. Bitcoin (Risk-On) mit Änderung
try:
    btc_data = yf.Ticker("BTC-USD").history(period="2d")
    btc_price = btc_data['Close'].iloc[-1]
    btc_prev = btc_data['Close'].iloc[-2]
    btc_delta = btc_price - btc_prev
    m_col4.metric("Bitcoin (Risk-On)", f"{btc_price:,.0f} $", f"{btc_delta:+.2f} $")
except:
    m_col4.error("BTC n.a.")

st.markdown("---")

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    puffer_limit = otm_puffer_slider / 100 
    
    with st.spinner("Lade Marktliste..."):
        ticker_liste = get_combined_watchlist()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    all_results = []
    
    for i, symbol in enumerate(ticker_liste):
        if i % 5 == 0 or i == len(ticker_liste)-1:
            progress_bar.progress((i + 1) / len(ticker_liste))
            status_text.text(f"Scanne {i+1}/{len(ticker_liste)}: {symbol}...")
        
        try:
            res = get_stock_data_full(symbol)
            if res[0] is None or not res[1]: continue
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            if not (min_stock_price <= price <= max_stock_price): continue
            if only_uptrend and not uptrend: continue
            
            # Datums-Logik (11-20 Tage)
            available_dates = [d for d in dates if 11 <= (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days <= 20]
            target_date = available_dates[-1] if available_dates else next((d for d in dates if (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days >= 11), None)
            if not target_date: continue

            tk = yf.Ticker(symbol)
            chain = tk.option_chain(target_date).puts
            max_strike = price * (1 - puffer_limit)
            secure_options = chain[chain['strike'] <= max_strike].sort_values('strike', ascending=False)
            
            if not secure_options.empty:
                best_opt = secure_options.iloc[0]
                tage = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days
                
                # --- NEUE FILTER-LOGIK GEGEN EINBUCHUNGEN ---
                is_safe = True
                
                # 1. Earnings Blocker (21 Tage Puffer)
                if earn:
                    try:
                        current_year = datetime.now().year
                        e_date = datetime.strptime(f"{earn}{current_year}", "%d.%m.%Y")
                        # Wenn Earnings innerhalb der nächsten (Tage + 3) liegen -> Aussortieren
                        if datetime.now() < e_date < (datetime.now() + timedelta(days=tage + 3)):
                            is_safe = False
                    except: pass

                # 2. RSI Überverkauft-Schutz (Kein Put-Verkauf bei Panik)
                if rsi < 35:
                    is_safe = False

                if is_safe:
                    bid = best_opt['bid'] if best_opt['bid'] > 0 else (best_opt['lastPrice'] if best_opt['lastPrice'] > 0 else 0.05)
                    y_pa = (bid / best_opt['strike']) * (365 / max(1, tage)) * 100
                    puffer_ist = ((price - best_opt['strike']) / price) * 100
                    
                    if y_pa >= min_yield_pa:
                        all_results.append({
                            'symbol': symbol, 'price': price, 'y_pa': y_pa, 'strike': best_opt['strike'],
                            'puffer': puffer_ist, 'bid': bid, 'rsi': rsi, 'uptrend': uptrend,
                            'earn': earn, 'tage': tage, 'date': target_date
                        })
        except: continue

    all_results = sorted(all_results, key=lambda x: x['y_pa'], reverse=True)
    status_text.empty()
    progress_bar.empty()
    
    if not all_results:
        st.warning("Keine Treffer gefunden.")
    else:
        st.success(f"Scan beendet. {len(all_results)} Chancen sortiert nach Rendite identifiziert!")
        cols = st.columns(4)
        for idx, res in enumerate(all_results):
            with cols[idx % 4]:
                earn_warning = f" ⚠️ <span style='color:#e67e22; font-size:0.8em;'>ER: {res['earn']}</span>" if res['earn'] else ""
                rsi_color = "#e74c3c" if res['rsi'] > 70 else "#2ecc71" if res['rsi'] < 40 else "#555"
                with st.container(border=True):
                    st.markdown(f"**{res['symbol']}** {'✅' if res['uptrend'] else '📉'}{earn_warning}", unsafe_allow_html=True)
                    st.metric("Yield p.a.", f"{res['y_pa']:.1f}%")
                    st.markdown(f"""
                    <div style="font-size: 0.85em; line-height: 1.4; background-color: #f1f3f6; padding: 10px; border-radius: 8px; border-left: 5px solid #2ecc71;">
                    <b style="color: #1e7e34; font-size: 1.1em;">Prämie: {res['bid']:.2f}$</b><br>
                    <span style="color: #666;">({res['bid']*100:.0f}$ pro Kontrakt)</span><hr style="margin: 8px 0;">
                    <b>Strike:</b> {res['strike']:.1f}$ ({res['puffer']:.1f}% Puffer)<br>
                    <b>Kurs:</b> {res['price']:.2f}$ | <b>RSI:</b> <span style="color:{rsi_color}; font-weight:bold;">{res['rsi']:.0f}</span><br>
                    <b>Termin:</b> {res['tage']} Tage
                    </div>
                    """, unsafe_allow_html=True)

# --- SEKTION 2: SMART DEPOT-MANAGER (REPAIR VERSION) ---
st.markdown("### 💼 Smart Depot-Manager (Aktiv)")
depot_data = [
    {'Ticker': 'AFRM', 'Einstand': 76.00}, {'Ticker': 'HOOD', 'Einstand': 120.0},
    {'Ticker': 'JKS', 'Einstand': 50.00}, {'Ticker': 'GTM', 'Einstand': 17.00},
    {'Ticker': 'HIMS', 'Einstand': 37.00}, {'Ticker': 'NVO', 'Einstand': 97.00},
    {'Ticker': 'RBRK', 'Einstand': 70.00}, {'Ticker': 'SE', 'Einstand': 170.00},
    {'Ticker': 'ETSY', 'Einstand': 67.00}, {'Ticker': 'TTD', 'Einstand': 102.00},
    {'Ticker': 'ELF', 'Einstand': 109.00}
]       

p_cols = st.columns(4) 
for i, item in enumerate(depot_data):
    price, _, earn, rsi, uptrend, _, _ = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        perf_color = "#2ecc71" if diff >= 0 else "#e74c3c"
        with p_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{item['Ticker']}** <span style='float:right; color:{perf_color}; font-weight:bold;'>{diff:+.1f}%</span>", unsafe_allow_html=True)
                
                # Strategie-Logik
                if diff < -20:
                    # Repair-Ansatz: Call 15% über aktuellem Kurs, egal wo der Einstand ist
                    repair_strike = price * 1.15
                    st.warning("🛠️ Repair-Modus")
                    st.caption(f"Call @{repair_strike:.1f}$ senkt Einstand.")
                    if rsi < 40: 
                        st.info("Wait: RSI zu tief")
                    else:
                        st.success(f"Prämie einsammeln!")
                elif rsi > 65:
                    st.success("🟢 Call-Chance!")
                else:
                    st.info("⏳ Seitwärts")
                
                if earn: st.warning(f"📅 ER: {earn}")

# --- SEKTION 3: EINZEL-CHECK (REPARIERT: KLAMMERN & BID-PREISE) ---
st.markdown("---")
st.subheader("🔍 Einzel-Check & Option-Chain")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    ticker_symbol = t_in.strip().upper()
    with st.spinner(f"Analysiere {ticker_symbol}..."):
        price, dates, earn, rsi_now, uptrend, near_lower, atr = get_stock_data_full(ticker_symbol)
        tk = yf.Ticker(ticker_symbol)
        hist_small = tk.history(period="5d")
        
        if price is not None and not hist_small.empty:
            rsi_series = calculate_rsi(hist_small['Close'])
            rsi_prev = rsi_series.iloc[-2] if len(rsi_series) > 1 else rsi_now
            rsi_rising = rsi_now > rsi_prev

            # 1. Earnings-Check
            if earn:
                st.warning(f"⚠️ **Earnings am {earn}**")
                if "09.02" in earn or "10.02" in earn:
                    st.error("🛑 ACHTUNG: Earnings stehen unmittelbar bevor! Short Puts jetzt extrem riskant.")

            # 2. Strategie-Ampel
            st.markdown("#### 🚥 Strategie-Ampel")
            a1, a2, a3 = st.columns(3)
            if mode == "put":
                if rsi_now < 30:
                    if rsi_rising:
                        a1.error("🟡 BODENBILDUNG")
                    else:
                        a1.error("🔴 MESSER FÄLLT")
                elif 30 <= rsi_now <= 45 and rsi_rising:
                    a1.success("🟢 GO: SHORT PUT")
                else:
                    a1.info("⚪ NEUTRAL")
            else:
                a1.info("📝 CALL-MODUS")

            a2.metric("Aktueller RSI", f"{rsi_now:.1f}", f"{rsi_now - rsi_prev:+.1f}")
            a3.metric("Kurs", f"{price:.2f}$")

            # 3. Option-Chain mit Bid-Fix
            if not dates:
                st.warning("Keine Optionen verfügbar.")
            else:
                d_sel = st.selectbox("Laufzeit wählen", dates)
                try:
                    opt_obj = tk.option_chain(d_sel)
                    chain = opt_obj.puts if mode == "put" else opt_obj.calls
                    expiry_dt = datetime.strptime(d_sel, '%Y-%m-%d')
                    days_to_expiry = max(1, (expiry_dt - datetime.now()).days)
                    T_val = days_to_expiry / 365
                    
                    st.write("---")
                    # Filtern für Übersicht
                    if mode == "put":
                        df_view = chain[chain['strike'] <= price * 1.05].sort_values('strike', ascending=False)
                    else:
                        df_view = chain[chain['strike'] >= price * 0.95].sort_values('strike', ascending=True)
                    
                    for _, opt in df_view.head(10).iterrows():
                        # Preis-Fix für Wochenende
                        display_bid = opt['bid'] if opt['bid'] > 0 else opt['lastPrice']
                        
                        # IV-Fix: Wir stellen sicher, dass iv_val eine kleine Dezimalzahl ist
                        iv_val = opt['impliedVolatility'] if opt['impliedVolatility'] else 0.4
                        
                        # Delta berechnen
                        calc_delta = calculate_bsm_delta(price, opt['strike'], T_val, iv_val, option_type=mode)
                        d_abs = abs(calc_delta)
                        
                        # Ampel-Logik (Sorgt für die bunten Punkte)
                        risk_emoji = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.35 else "🔴"
                        
                        # Puffer berechnen (Fehler-Fix für Bild 2)
                        puffer = (abs(opt['strike'] - price) / price) * 100
                        
                        # Rendite p.a.
                        y_pa = (display_bid / opt['strike']) * (365 / days_to_expiry) * 100 if display_bid > 0 else 0
                        
                        st.markdown(
                            f"{risk_emoji} **Strike: {opt['strike']:.1f}** | "
                            f"Bid (Last): {display_bid:.2f}$ | "
                            f"Delta: {d_abs:.2f} | "
                            f"Puffer: {puffer:.1f}% | "
                            f"Yield: {y_pa:.1f}% p.a.",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"Fehler in Chain: {e}")




