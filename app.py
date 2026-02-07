import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- 1. SETUP & ROBUSTE MATHE ---
st.set_page_config(page_title="CapTrader AI Market Guard Pro", layout="wide")

def calculate_bsm_delta(S, K, T, sigma, r=0.04, option_type='put'):
    """Berechnet Delta mit hartem Fallback gegen 0.00 Werte."""
    T = max(T, 0.0001)
    # Wenn Vola fehlt oder unrealistisch niedrig ist, nimm 45% als Sicherheitsanker
    sig = sigma if (sigma and sigma > 0.05) else 0.45
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        if option_type == 'call':
            delta = float(norm.cdf(d1))
        else:
            delta = float(norm.cdf(d1) - 1)
        return round(delta, 2)
    except:
        return -0.40 if option_type == 'put' else 0.40 # Grober Schätzwert als Notfall

@st.cache_data(ttl=600)
def get_stock_basics(symbol):
    """Sicherer Datenabruf für Ticker-Stammdaten."""
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        earn_date = ""
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                earn_date = cal['Earnings Date'][0].strftime('%d.%m.')
        except: pass
        return price, dates, earn_date
    except:
        return None, [], ""

# --- 2. SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Filter")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 85)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Guard Pro")

# --- 3. MARKT-SCANNER (FIX FÜR LARGEUTF8 FEHLER) ---
if st.button("🚀 High-Safety Scan starten"):
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "COIN", "PLTR", "HOOD", "MSTR", "UBER", "DIS", "PYPL", "AFRM", "SQ", "RIVN"]
    results = []
    prog = st.progress(0)
    
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn = get_stock_basics(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                # Nimm die Laufzeit, die am nächsten an 30 Tagen ist
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                
                days = max((datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days, 1)
                T = days / 365
                
                # Delta berechnen
                chain['delta_val'] = chain.apply(lambda r: calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility']), axis=1)
                
                # Nur Puts im Delta-Bereich mit Gebot
                matches = chain[(chain['delta_val'].abs() <= max_delta) & (chain['bid'] > 0)].copy()
                
                if not matches.empty:
                    matches['y_pa'] = (matches['bid'] / matches['strike']) * (365 / days) * 100
                    best = matches.sort_values('y_pa', ascending=False).iloc[0]
                    
                    if best['y_pa'] >= min_yield_pa:
                        # WICHTIG: Nur einfache Datentypen für die Tabelle (verhindert LargeUtf8 Error)
                        results.append({
                            "Ticker": str(t),
                            "Rendite": f"{float(best['y_pa']):.1f}%",
                            "Strike": float(best['strike']),
                            "Kurs": f"{float(price):.2f}$",
                            "Earnings": str(earn)
                        })
            except:
                continue
                
    if results:
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True) # dataframe ist stabiler als table
    else:
        st.warning("Keine Treffer gefunden. Erhöhe das Risiko (Delta) oder senke die Rendite.")

st.markdown("---")

# --- 4. DEPOT-STATUS (ALLE 12 WERTE) ---
st.subheader("💼 Depot-Status")
depot_list = ["AFRM", "ELF", "ETSY", "GTLB", "GTM", "HIMS", "HOOD", "JKS", "NVO", "RBRK", "SE", "TTD"]
d_cols = st.columns(4)

for i, t in enumerate(depot_list):
    price, _, earn = get_stock_basics(t)
    if price:
        with d_cols[i % 4]:
            st.metric(t, f"{price:.2f}$")
            if earn: st.caption(f"Earnings: {earn}")

st.markdown("---")

# --- 5. EINZEL-CHECK ---
st.subheader("🔍 Experten Einzel-Check")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Optionstyp", ["put", "call"], horizontal=True, key="check_mode")
with c2: t_in = st.text_input("Ticker Symbol", value="ELF").upper().strip()

if t_in:
    price, dates, earn = get_stock_basics(t_in)
    if price and dates:
        st.write(f"Aktueller Kurs: **{price:.2f}$**")
        d_sel = st.selectbox("Laufzeit wählen", dates, key=f"sb_{t_in}")
        
        try:
            tk = yf.Ticker(t_in)
            chain = tk.option_chain(d_sel)
            df = chain.puts if mode == "put" else chain.calls
            T = max((datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days, 1) / 365
            
            # Strike-Sortierung: Nah am Geld zuerst
            if mode == "put":
                df = df[df['strike'] <= price * 1.02].sort_values('strike', ascending=False)
            else:
                df = df[df['strike'] >= price * 0.98].sort_values('strike', ascending=True)
            
            for _, opt in df.head(8).iterrows():
                delta = calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'], mode)
                d_abs = abs(delta)
                
                # Risiko-Status
                is_itm = (mode == "put" and opt['strike'] > price) or (mode == "call" and opt['strike'] < price)
                status = "🔴 ITM" if is_itm else "🟢 OTM" if d_abs < 0.25 else "🟡 RISK"
                
                with st.expander(f"{status} | Strike {opt['strike']:.1f}$ | Bid: {opt['bid']:.2f}$ | Delta: {d_abs:.2f}"):
                    cola, colb = st.columns(2)
                    with cola:
                        st.write(f"📊 **OTM-Chance:** {(1-d_abs)*100:.1f}%")
                        st.write(f"💰 **Prämie:** {opt['bid']*100:.0f}$")
                    with colb:
                        st.write(f"🎯 **Puffer:** {(abs(opt['strike']-price)/price)*100:.1f}%")
                        st.write(f"💼 **Kapital:** {opt['strike']*100:,.0f}$")
        except:
            st.error("Optionskette konnte nicht geladen werden.")
