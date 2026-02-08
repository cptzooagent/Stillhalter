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
@st.cache_data(ttl=3600)
def get_combined_watchlist():
    sp500_nasdaq_mix = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "NFLX", 
        "AMD", "INTC", "QCOM", "AMAT", "TXN", "MU", "ISRG", "LRCX", "PANW", "SNPS",
        "LLY", "V", "MA", "JPM", "WMT", "XOM", "UNH", "PG", "ORCL", "COST", 
        "ABBV", "BAC", "KO", "PEP", "CRM", "WFC", "DIS", "CAT", "AXP", "IBM",
        "COIN", "MARA", "PLTR", "AFRM", "SQ", "RIVN", "UPST", "HOOD", "SOFI", "MSTR"
    ]
    return list(set(sp500_nasdaq_mix))

@st.cache_data(ttl=900)
@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        # Basis-Daten
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        
        # Historie für Indikatoren (6 Monate für SMA 200)
        hist = tk.history(period="150d") # 150 Handelstage (~7 Monate)
        
        if not hist.empty and len(hist) > 20:
            # 1. RSI
            rsi_val = calculate_rsi(hist['Close']).iloc[-1]
            
            # 2. SMA 200 (Trend-Check) - wir nehmen hier den verfügbaren Max-Zeitraum
            sma_200 = hist['Close'].mean() 
            is_uptrend = price > sma_200
            
            # 3. Bollinger Bänder (20 Tage, 2 Standardabweichungen)
            sma_20 = hist['Close'].rolling(window=20).mean()
            std_20 = hist['Close'].rolling(window=20).std()
            lower_band = (sma_20 - 2 * std_20).iloc[-1]
            is_near_lower = price <= (lower_band * 1.02) # Innerhalb 2% vom Band
            
            # 4. ATR (einfache Version für die Vola)
            high_low = hist['High'] - hist['Low']
            atr = high_low.rolling(window=14).mean().iloc[-1]
        else:
            rsi_val, is_uptrend, is_near_lower, atr = 50, True, False, 0
            
        # Earnings
        earn_str = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_str = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        
        return price, dates, earn_str, rsi_val, is_uptrend, is_near_lower, atr
    except:
        return None, [], "", 50, True, False, 0

# --- UI: SEITENLEISTE ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
sort_by_rsi = st.sidebar.checkbox("Nach RSI sortieren (Hoch -> Tief)")
# --- NEUER SCHIEBER FÜR MINDESTPREIS ---
min_stock_price = st.sidebar.slider("Mindest-Aktienpreis ($)", 0, 500, 20)

# --- HAUPTBEREICH ---
st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: PROFI-EINSTIEGS-CHANCEN (ULTRA-FINDER) ---
st.subheader("🎯 Profi-Einstiegs-Chancen")

if st.button("🚀 Kombi-Scan starten"):
    # Ticker-Liste (erweitert)
    ticker_liste = ["HOOD", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "CCJ"]
    cols = st.columns(4)
    col_idx = 0
    
    with st.spinner("Scanne Märkte..."):
        for symbol in ticker_liste:
            try:
                # 1. Kursdaten holen
                res = get_stock_data_full(symbol)
                if not res or res[0] is None: continue
                price, dates, earn, rsi, uptrend, near_lower, atr = res
                
                # 2. Options-Check
                if not dates: continue
                tk = yf.Ticker(symbol)
                d_sel = dates[0] # Nächste Laufzeit
                chain = tk.option_chain(d_sel).puts
                
                # Zeit bis Expiry
                expiry_dt = datetime.strptime(d_sel, '%Y-%m-%d')
                tage = max(1, (expiry_dt - datetime.now()).days)
                
                # Delta berechnen (ca. 0.16 suchen)
                chain['delta_calc'] = chain.apply(lambda o: calculate_bsm_delta(
                    price, o['strike'], tage/365, o['impliedVolatility'] or 0.4, "put"
                ), axis=1)
                
                # Den sichersten Strike finden
                best_opt = chain.iloc[(chain['delta_calc'] + 0.16).abs().argsort()[:1]].iloc[0]
                
                # Werte für die Anzeige
                bid = best_opt['bid'] if best_opt['bid'] > 0 else (best_opt['lastPrice'] or 0.1)
                y_pa = (bid / best_opt['strike']) * (365 / tage) * 100
                puffer = (abs(best_opt['strike'] - price) / price) * 100
                fmt_date = expiry_dt.strftime('%d.%m.')

                # 3. Karte rendern
                with cols[col_idx % 4]:
                    with st.container(border=True):
                        st.markdown(f"### 🟡 {symbol}")
                        st.caption(f"🗓️ ER: {earn if earn else 'N/A'}")
                        st.write("Yield p.a.")
                        # Große Anzeige der Rendite
                        st.title(f"{y_pa:.1f}%")
                        st.markdown(f"**Strike: {best_opt['strike']:.1f}$**")
                        st.markdown(f"**Laufzeit: {fmt_date}**")
                        st.caption(f"Puffer: {puffer:.1f}% | RSI: {rsi:.0f}")
                
                col_idx += 1
                
            except Exception:
                continue # Wenn ein Symbol hakt, einfach ignorieren und weiter

    if col_idx == 0:
        st.warning("Yahoo Timeout oder keine Daten. Bitte noch einmal klicken.")

# Beispiel-Daten für dein Depot (Hier deine echten Werte eintragen!)
depot_data = [
    {'Ticker': 'AFRM', 'Einstand': 76.00},
    {'Ticker': 'HOOD', 'Einstand': 82.82},
    {'Ticker': 'JKS', 'Einstand': 50.00},
    {'Ticker': 'GTM', 'Einstand': 17.00},
    {'Ticker': 'HIMS', 'Einstand': 37.00},
    {'Ticker': 'ETSY', 'Einstand': 67.00},
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






