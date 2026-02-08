import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import engine as eng # Importiert die engine.py

# --- SETUP ---
st.set_page_config(page_title="CapTrader AI Market Scanner", layout="wide")

@st.cache_data(ttl=900)
def get_stock_data_full(symbol):
    try:
        tk = yf.Ticker(symbol)
        price = tk.fast_info['last_price']
        dates = list(tk.options)
        hist = tk.history(period="1mo")
        rsi_val = eng.calculate_rsi(hist['Close']).iloc[-1] if not hist.empty else 50
        earn_dt = eng.get_clean_earnings(tk)
        earn_str = earn_dt.strftime('%d.%m.') if earn_dt else ""
        return price, dates, earn_str, rsi_val, earn_dt
    except:
        return None, [], "", 50, None

# --- SIDEBAR ---
st.sidebar.header("🛡️ Strategie-Einstellungen")
target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 98, 83)
max_delta = (100 - target_prob) / 100
min_yield_pa = st.sidebar.number_input("Mindestrendite p.a. (%)", value=20)
sort_by_rsi = st.sidebar.checkbox("Nach RSI sortieren (Günstiger Einstieg)")

st.title("🛡️ CapTrader AI Market Scanner")

# --- SEKTION 1: KOMBI-SCAN ---
if st.button("🚀 Kombi-Scan starten"):
    watchlist = eng.get_watchlist()
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(watchlist):
        status.text(f"Analysiere {t}...")
        prog.progress((i + 1) / len(watchlist))
        price, dates, earn, rsi, _ = get_stock_data_full(t)
        
        if price and dates:
            try:
                tk = yf.Ticker(t)
                target_date = min(dates, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days - 30))
                chain = tk.option_chain(target_date).puts
                T = (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days / 365
                
                chain['delta_val'] = chain.apply(lambda r: eng.calculate_bsm_delta(price, r['strike'], T, r['impliedVolatility'] or 0.4), axis=1)
                safe_opts = chain[chain['delta_val'].abs() <= max_delta].copy()
                
                if not safe_opts.empty:
                    days = max(1, (datetime.strptime(target_date, '%Y-%m-%d') - datetime.now()).days)
                    safe_opts['y_pa'] = (safe_opts['bid'] / safe_opts['strike']) * (365 / days) * 100
                    matches = safe_opts[safe_opts['y_pa'] >= min_yield_pa]
                    
                    if not matches.empty:
                        best = matches.sort_values('y_pa', ascending=False).iloc[0]
                        results.append({
                            'ticker': t, 'yield': best['y_pa'], 'strike': best['strike'], 
                            'bid': best['bid'], 'puffer': (abs(best['strike'] - price) / price) * 100, 
                            'delta': abs(best['delta_val']), 'earn': earn, 'rsi': rsi
                        })
            except: continue

    status.text("Kombinations-Scan abgeschlossen!")
    if results:
        df_res = pd.DataFrame(results).sort_values('rsi' if sort_by_rsi else 'yield', ascending=not sort_by_rsi)
        cols = st.columns(4)
        for idx, row in enumerate(df_res.head(12).to_dict('records')):
            with cols[idx % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"💰 Bid: **{row['bid']:.2f}$** | Puffer: **{row['puffer']:.1f}%**")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$** (Δ {row['delta']:.2f})")
                if row['earn']: st.warning(f"⚠️ ER: {row['earn']}")
    else:
        st.warning("Keine Treffer.")

st.divider()

# --- SEKTION 2: SMART DEPOT-MANAGER ---
st.subheader("💼 Smart Depot-Manager")
depot_data = [
    {"Ticker": "AFRM", "Einstand": 76.0}, {"Ticker": "HOOD", "Einstand": 82.82},
    {"Ticker": "PLTR", "Einstand": 25.0}, {"Ticker": "MSTR", "Einstand": 1500.0}
]

p_cols = st.columns(3)
for i, item in enumerate(depot_data):
    price, _, earn, rsi, earn_dt = get_stock_data_full(item['Ticker'])
    if price:
        diff = (price / item['Einstand'] - 1) * 100
        with p_cols[i % 3]:
            with st.expander(f"{item['Ticker']} ({diff:.1f}%)", expanded=True):
                if earn_dt is not None:
                    try:
                        now = datetime.now().replace(tzinfo=None)
                        days_to_earn = (earn_dt.replace(tzinfo=None) - now).days
                        if 0 <= days_to_earn <= 7:
                            st.error(f"🚨 EARNINGS IN {days_to_earn} TAGEN!")
                    except: pass
                st.metric("Kurs", f"{price:.2f}$", f"{diff:.1f}%")
                st.caption(f"RSI: {rsi:.0f}")
                if rsi > 65: st.success("✅ Call-Verkauf prüfen")
                if rsi < 35: st.info("💎 Oversold - Hold")

st.divider()

# --- SEKTION 3: EINZEL-CHECK (AMPELSYSTEM) ---
st.subheader("🔍 Einzel-Check & Option-Chain")
c1, c2 = st.columns([1, 2])
with c1: mode = st.radio("Optionstyp", ["put", "call"], horizontal=True)
with c2: t_in = st.text_input("Ticker Symbol", value="HOOD").upper()

if t_in:
    price, dates, earn, rsi, _ = get_stock_data_full(t_in)
    if price and dates:
        st.info(f"Aktueller Kurs: **{price:.2f}$** | RSI: **{rsi:.0f}** | Earnings: {earn}")
        d_sel = st.selectbox("Laufzeit wählen", dates)
        tk = yf.Ticker(t_in)
        
        try:
            chain = tk.option_chain(d_sel).puts if mode == "put" else tk.option_chain(d_sel).calls
            T = (datetime.strptime(d_sel, '%Y-%m-%d') - datetime.now()).days / 365
            
            chain['delta_calc'] = chain.apply(lambda opt: eng.calculate_bsm_delta(
                price, opt['strike'], T, opt['impliedVolatility'] or 0.4, option_type=mode
            ), axis=1)

            if mode == "put":
                filtered_df = chain[chain['strike'] <= price * 1.05].sort_values('strike', ascending=False)
            else:
                filtered_df = chain[chain['strike'] >= price * 0.95].sort_values('strike', ascending=True)
            
            for _, opt in filtered_df.head(15).iterrows():
                d_abs = abs(opt['delta_calc'])
                # AMPEL LOGIK
                color = "🟢" if d_abs < 0.16 else "🟡" if d_abs <= 0.30 else "🔴"
                y_pa = (opt['bid'] / opt['strike']) * (365 / max(1, T*365)) * 100
                
                st.markdown(
                    f"{color} **Strike: {opt['strike']:.1f}$** | "
                    f"Bid: <span style='color:#2ecc71;'>{opt['bid']:.2f}$</span> | "
                    f"Delta: {d_abs:.2f} | "
                    f"Puffer: {abs(opt['strike']-price)/price*100:.1f}% | "
                    f"Yield: {y_pa:.1f}% p.a.", 
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Fehler: {e}")
