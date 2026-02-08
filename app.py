import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import engine as eng

st.set_page_config(page_title="CapTrader AI Scanner", layout="wide")

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = eng.calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_dt = eng.get_clean_earnings(tk)
        earn_str = earn_dt.strftime('%d.%m.') if earn_dt else "N/A"
        return price, dates, earn_str, rsi_val, earn_dt
    except:
        return None, [], "", 50, None

# --- SIDEBAR ---
st.sidebar.header("🛡️ Strategie")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: SCANNER ---
if st.button("🚀 Markt-Scan starten"):
    results = []
    watchlist = eng.get_watchlist()
    prog = st.progress(0)
    for i, t in enumerate(watchlist):
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        if price and dates:
            try:
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                tk = yf.Ticker(t)
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                chain['delta'] = chain.apply(lambda r: eng.calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                matches = chain[(chain['delta'].abs() <= max_delta) & (chain['bid'] > 0.05)]
                if not matches.empty:
                    best = matches.sort_values('strike', ascending=False).iloc[0]
                    y_pa = (best['bid'] / best['strike']) * (365 / max(1, T*365)) * 100
                    if y_pa >= min_yield_pa:
                        results.append({'Ticker': t, 'Rendite': y_pa, 'Strike': best['strike'], 'Delta': abs(best['delta']), 'RSI': rsi})
            except: continue
    if results:
        st.table(pd.DataFrame(results))

st.divider()

# --- SEKTION 2: DEPOT ---
st.header("💼 Smart Depot-Manager")
depot = [{"T": "AFRM", "E": 76.0}, {"T": "HOOD", "E": 82.82}, {"T": "PLTR", "E": 25.0}]
cols = st.columns(len(depot))
for idx, item in enumerate(depot):
    price, _, earn, rsi, _ = get_stock_data_full(item['T'])
    if price:
        diff = (price/item['E']-1)*100
        cols[idx].metric(item['T'], f"{price:.2f}$", f"{diff:.1f}%")
        cols[idx].caption(f"RSI: {rsi:.0f} | ER: {earn}")

st.divider()

# --- SEKTION 3: EINZEL-CHECK (DEIN SCREENSHOT SYSTEM) ---
st.header("🔍 Einzel-Check & Option-Chain")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Typ", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        st.info(f"Aktueller Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}** | Nächste Ernte: {earn}")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        
        try:
            tk = yf.Ticker(t_in)
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            # Berechnungen
            chain['delta_calc'] = chain.apply(lambda opt: eng.calculate_bsm_delta(price, opt['strike'], T, opt['impliedVolatility'] or 0.4, mode), axis=1)
            
            # Filterung: Zeige Strikes um den aktuellen Kurs
            if mode == "put":
                filtered = chain[chain['strike'] <= price * 1.1].sort_values('strike', ascending=False)
            else:
                filtered = chain[chain['strike'] >= price * 0.9].sort_values('strike', ascending=True)

            st.write("---")
            for _, opt in filtered.head(20).iterrows():
                d_abs = abs(opt['delta_calc'])
                # AMPEL
                color = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.30 else "🔴"
                y_pa = (opt['bid'] / opt['strike']) * (365 / max(1, T*365)) * 100
                puffer = (abs(opt['strike'] - price) / price) * 100
                
                # HTML Formatierung für die grüne Prämie
                st.markdown(
                    f"{color} **Strike: {opt['strike']:.1f}** | "
                    f"Bid: <span style='color:#2ecc71;'>{opt['bid']:.2f}$</span> | "
                    f"Delta: {d_abs:.2f} | "
                    f"Puffer: {puffer:.1f}% | "
                    f"Rendite: **{y_pa:.1f}% p.a.**", 
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Fehler: {e}")
