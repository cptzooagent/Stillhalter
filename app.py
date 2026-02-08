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
    if len(data) < window + 1: return 50
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 2. DATEN-FUNKTIONEN ---
@st.cache_data(ttl=86400)
def get_combined_watchlist():
    try:
        # Stabilere Quelle für S&P 500 (GitHub CSV statt Wikipedia)
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        
        # Manuelle Ergänzung wichtiger Nasdaq/Wachstumswerte, falls sie im S&P fehlen
        nasdaq_extra = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR", "HOOD", "PLTR", "SQ"]
        
        full_list = list(set(tickers + nasdaq_extra))
        # yfinance braucht Bindestriche statt Punkte (z.B. BRK-B)
        return [t.replace('.', '-') for t in full_list]
    except Exception as e:
        # Dein Sicherheitsnetz, falls der Download scheitert
        return ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "COIN", "MSTR"]

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        # Sicherere Methode: Wir ziehen die Historie, um den letzten Preis zu bekommen
        hist = tk.history(period="150d") 
        if hist.empty: return None, [], "", 50, True, False, 0
            
        price = hist['Close'].iloc[-1] 
        dates = list(tk.options)
        
        # Indikatoren
        rsi_val = calculate_rsi(hist['Close']).iloc[-1]
        sma_200 = hist['Close'].mean() 
        is_uptrend = price > sma_200
        
        # Bollinger & ATR
        sma_20 = hist['Close'].rolling(window=20).mean()
        std_20 = hist['Close'].rolling(window=20).std()
        lower_band = (sma_20 - 2 * std_20).iloc[-1]
        is_near_lower = price <= (lower_band * 1.02)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
            
        # Earnings-Check (stabilisiert)
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

# --- UI: SEITENLEISTE (CLEAN VERSION) ---
st.sidebar.header("🛡️ Strategie-Einstellungen")

# 1. Puffer-Steuerung
otm_puffer_slider = st.sidebar.slider(
    "Gewünschter Puffer (%)", 
    min_value=3, 
    max_value=25, 
    value=10,
    help="Wie weit muss der Strike unter dem aktuellen Kurs liegen?"
)

# 2. Rendite-Steuerung
min_yield_pa = st.sidebar.number_input(
    "Mindestrendite p.a. (%)", 
    min_value=0, 
    max_value=100, 
    value=15
)

# 3. Aktienpreis-Filter
min_stock_price = st.sidebar.slider(
    "Mindest-Aktienpreis ($)", 
    0, 500, 20
)

# 4. Strategie-Filter
st.sidebar.markdown("---")
only_uptrend = st.sidebar.checkbox("Nur Aufwärtstrend (SMA 200)", value=False)
st.sidebar.info("Tipp: Deaktiviere den Aufwärtstrend für mehr Treffer am Wochenende.")

# Konstante für Delta-Berechnungen im Hintergrund
max_delta = 0.20

if st.button("🚀 Kombi-Scan starten"):
    puffer_limit = otm_puffer_slider / 100 
    
    # Holt die dynamische Liste (S&P 500 + Nasdaq)
    with st.spinner("Lade aktuelle Marktliste..."):
        ticker_liste = get_combined_watchlist()
    
    st.info(f"Suche in {len(ticker_liste)} Symbolen nach Puts mit >{otm_puffer_slider}% Puffer und >{min_yield_pa}% Rendite...")
    
    cols = st.columns(4)
    found_idx = 0
    
    # Status-Elemente für den Benutzer
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Der eigentliche Scan-Loop
    for i, symbol in enumerate(ticker_liste):
        # Update Fortschritt (alle 5 Ticker für bessere Performance)
        if i % 5 == 0 or i == len(ticker_liste)-1:
            progress_bar.progress((i + 1) / len(ticker_liste))
            status_text.text(f"Analysiere {i+1}/{len(ticker_liste)}: {symbol}...")
        
        try:
            # 1. Basis-Daten holen
            res = get_stock_data_full(symbol)
            if res[0] is None or not res[1]: 
                continue
            
            price, dates, earn, rsi, uptrend, near_lower, atr = res
            
            # --- STRATEGIE-FILTER ---
            if price < min_stock_price: 
                continue
            if only_uptrend and not uptrend: 
                continue
            
            # 2. Passende Laufzeit finden (mind. 11 Tage)
            target_date = next((d for d in dates if (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days >= 11), None)
            if not target_date: 
                continue

            # 3. Optionskette laden
            tk = yf.Ticker(symbol)
            chain = tk.option_chain(target_date).puts
            
            # 4. Puffer-Check: Finde den besten Strike UNTER dem Limit
            max_strike = price * (1 - puffer_limit)
            secure_options = chain[chain['strike'] <= max_strike].sort_values('strike', ascending=False)
            
            if secure_options.empty: 
                continue
            
            best_opt = secure_options.iloc[0]
            
            # 5. Rendite-Check (Wochenend-sicher)
            bid = best_opt['bid'] if best_opt['bid'] > 0 else (best_opt['lastPrice'] if best_opt['lastPrice'] > 0 else 0.05)
            
            expiry_dt = datetime.strptime(target_date, '%Y-%m-%d')
            tage = (expiry_dt - datetime.now()).days
            y_pa = (bid / best_opt['strike']) * (365 / max(1, tage)) * 100
            puffer_ist = ((price - best_opt['strike']) / price) * 100
            
            if y_pa < min_yield_pa: 
                continue

            # --- ANZEIGE IN 4 SPALTEN ---
            with cols[found_idx % 4]:
                # RSI-Ampel Logik
                rsi_color = "#e74c3c" if rsi > 70 else "#2ecc71" if rsi < 40 else "#555"
                rsi_weight = "bold" if rsi > 70 or rsi < 40 else "normal"
                
                with st.container(border=True):
                    # Header: Symbol und Trend-Status
                    st.markdown(f"**{symbol}** {'✅' if uptrend else '📉'}")
                    st.metric("Yield p.a.", f"{y_pa:.1f}%")
                    
                    # Die Info-Box mit Prämie und RSI-Warnung
                    st.markdown(f"""
                    <div style="font-size: 0.85em; line-height: 1.4; background-color: #f1f3f6; padding: 10px; border-radius: 8px; border-left: 5px solid #2ecc71;">
                    <b style="color: #1e7e34; font-size: 1.1em;">Prämie: {bid:.2f}$</b><br>
                    <span style="color: #666;">(Einnahme: {bid*100:.0f}$ pro Kontrakt)</span><hr style="margin: 8px 0;">
                    <b>Strike:</b> {best_opt['strike']:.1f}$ ({puffer_ist:.1f}% Puffer)<br>
                    <b>Kurs:</b> {price:.2f}$ | <b>RSI:</b> <span style="color:{rsi_color}; font-weight:{rsi_weight};">{rsi:.0f}</span><br>
                    <b>Datum:</b> {expiry_dt.strftime('%d.%m.')} ({tage} Tage)
                    </div>
                    """, unsafe_allow_html=True)
            
            found_idx += 1

        except Exception as e:
            continue

    # Abschluss-Meldung
    status_text.empty()
    progress_bar.empty()
    if found_idx == 0:
        st.warning("Scan beendet. Keine Treffer gefunden. Tipp: Puffer oder Rendite-Anspruch senken.")
    else:
        st.success(f"Scan beendet. {found_idx} Chancen identifiziert!")
        
# Beispiel-Daten für dein Depot (Hier deine echten Werte eintragen!)
depot_data = [
    {'Ticker': 'AFRM', 'Einstand': 76.00},
    {'Ticker': 'HOOD', 'Einstand': 120.0},
    {'Ticker': 'JKS', 'Einstand': 50.00},
    {'Ticker': 'GTM', 'Einstand': 17.00},
    {'Ticker': 'HIMS', 'Einstand': 37.00},
    {'Ticker': 'NVO', 'Einstand': 97.00},
    {'Ticker': 'RBRK', 'Einstand': 70.00},
    {'Ticker': 'SE', 'Einstand': 170.00},
    {'Ticker': 'ETSY', 'Einstand': 67.00},
    {'Ticker': 'TTD', 'Einstand': 102.00},
    {'Ticker': 'ELF', 'Einstand': 109.00}
]       

# --- SEKTION 2: SMART DEPOT-MANAGER (FINAL) ---
st.markdown("### 💼 Smart Depot-Manager")

if 'depot_data' in locals():
    p_cols = st.columns(4) 
    for i, item in enumerate(depot_data):
        price, _, earn, rsi, uptrend, near_lower, atr = get_stock_data_full(item['Ticker'])
        
        if price:
            diff = (price / item['Einstand'] - 1) * 100
            perf_color = "#2ecc71" if diff >= 0 else "#e74c3c"
            
            with p_cols[i % 4]:
                with st.container(border=True):
                    # Header mit korrektem HTML
                    t_emoji = "📈" if uptrend else "📉"
                    st.markdown(
                        f"**{item['Ticker']}** {t_emoji} "
                        f"<span style='float:right; color:{perf_color}; font-weight:bold;'>{diff:+.1f}%</span>", 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"<p style='font-size:13px; margin:0;'>Kurs: {price:.2f}$ | RSI: {rsi:.0f}</p>", unsafe_allow_html=True)
                    
                    # --- CALL-STRATEGIE ---
                    if diff < -15:
                        st.error("⚠️ Call-Gefahr!")
                        st.caption(f"Einstand {item['Einstand']}$ zu weit weg.")
                    elif rsi > 60:
                        st.success("🟢 Call-Chance!")
                        st.caption("RSI heiß. Jetzt Calls prüfen.")
                    else:
                        st.info("⏳ Warten")
                        st.caption(f"Target: RSI > 60")

                    # Earnings-Check
                    if earn:
                        st.warning(f"📅 ER: {earn}")
else:
    st.error("Variable 'depot_data' wurde nicht gefunden!")

# --- SEKTION 3: EINZEL-CHECK (STABILE AMPEL-VERSION) ---
st.subheader("🔍 Einzel-Check & Option-Chain")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    ticker_symbol = t_in.strip().upper()
    
    with st.spinner(f"Analysiere {ticker_symbol}..."):
        # Sicherstellen, dass die Funktion 7 Werte liefert
        price, dates, earn, rsi, uptrend, near_lower, atr = get_stock_data_full(ticker_symbol)
    
    if price is None:
        st.error(f"❌ Keine Daten für '{ticker_symbol}' gefunden.")
    elif not dates:
        st.warning(f"⚠️ Keine Optionen für {ticker_symbol} verfügbar.")
    else:
        # 1. Dashboard-Header
        h1, h2, h3 = st.columns(3)
        h1.metric("Kurs", f"{price:.2f}$")
        h2.metric("Trend", "📈 Bullisch" if uptrend else "📉 Bärisch")
        h3.metric("RSI", f"{rsi:.0f}")

        if not uptrend:
            st.error("🛑 Achtung: Aktie notiert unter SMA 200 (Abwärtstrend)!")
        
        d_sel = st.selectbox("Laufzeit wählen", dates)
        
        try:
            tk = yf.Ticker(ticker_symbol)
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            
            expiry_dt = datetime.strptime(d_sel, '%Y-%m-%d')
            days_to_expiry = max(1, (expiry_dt - datetime.now()).days)
            T = days_to_expiry / 365
            
            # Delta-Berechnung mit Absicherung gegen fehlende Vola (None)
            chain['delta_calc'] = chain.apply(lambda opt: calculate_bsm_delta(
                price, opt['strike'], T, (opt['impliedVolatility'] if opt['impliedVolatility'] else 0.4), option_type=mode
            ), axis=1)

            if mode == "put":
                filtered_df = chain[chain['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                filtered_df = chain[chain['strike'] >= price * 0.95].sort_values('strike', ascending=True)
            
            st.write("---")
            for _, opt in filtered_df.head(15).iterrows():
                # Sicherheits-Check: Falls Bid fehlt (NaN), auf 0 setzen
                bid_val = opt['bid'] if not pd.isna(opt['bid']) else 0.0
                d_abs = abs(opt['delta_calc'])
                
                # Ampel-Logik
                risk_emoji = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.30 else "🔴"
                
                # Rendite & Puffer
                y_pa = (bid_val / opt['strike']) * (365 / days_to_expiry) * 100
                puffer = (abs(opt['strike'] - price) / price) * 100
                
                # Das funktionierende Design
                bid_style = f"<span style='color:#2ecc71; font-weight:bold;'>{bid_val:.2f}$</span>"
                
                st.markdown(
                    f"{risk_emoji} **Strike: {opt['strike']:.1f}** | "
                    f"Bid: {bid_style} | "
                    f"Delta: {d_abs:.2f} | "
                    f"Puffer: {puffer:.1f}% | "
                    f"Yield: {y_pa:.1f}% p.a.",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Fehler bei der Anzeige: {e}")
# --- ENDE DER DATEI ---



