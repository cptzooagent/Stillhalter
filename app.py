import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Repair Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- DATA FUNCTIONS ---
def get_market_overview():
    data = {}
    vix_p = 0.0
    for ticker in ["VIX", "^VIX", "VXX"]:
        try:
            r = requests.get(f"https://api.marketdata.app/v1/indices/quotes/{ticker}/?token={MD_KEY}").json()
            if r.get('s') == 'ok' and r['last'][0] > 0:
                vix_p = r['last'][0]
                break
        except: continue
    data["VIX"] = {"price": vix_p if vix_p > 0 else 15.0}
    
    for name, etf in [("S&P 500", "SPY"), ("Nasdaq", "QQQ")]:
        try:
            r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={etf}&token={FINNHUB_KEY}').json()
            data[name] = {"price": r.get('c', 0.0) * (10 if name=="S&P 500" else 40), "change": r.get('dp', 0.0)}
        except: data[name] = {"price": 0.0, "change": 0.0}
    return data

def get_live_price(symbol):
    try:
        r = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}').json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_all_expirations(symbol):
    try:
        r = requests.get(f"https://api.marketdata.app/v1/options/expirations/{symbol}/?token={MD_KEY}").json()
        return sorted(r.get('expirations', [])) if r.get('s') == 'ok' else []
    except: return []

def get_chain_for_date(symbol, date_str, side):
    try:
        params = {"token": MD_KEY, "side": side, "expiration": date_str}
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{symbol}/", params=params).json()
        if r.get('s') == 'ok':
            return pd.DataFrame({'strike': r['strike'], 'mid': r['mid'], 'delta': r.get('delta', [0.0]*len(r['strike']))})
    except: return None

# --- UI START ---
st.title("🛡️ Stillhalter Repair Scanner")

market = get_market_overview()
vix_val = market["VIX"]["price"]

with st.container(border=True):
    m_cols = st.columns(3)
    m_cols[0].metric("VIX (Angst)", f"{vix_val:.2f}", "🔥 Panik" if vix_val > 25 else "🟢 Ruhig")
    m_cols[1].metric("S&P 500", f"{market['S&P 500']['price']:,.0f}")
    m_cols[2].metric("Nasdaq", f"{market['Nasdaq']['price']:,.0f}")

st.divider()

# --- PORTFOLIO & REPAIR AMPEL ---
st.subheader("💼 Portfolio Repair-Status")
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "AFRM", "Einstand": 76.00}, {"Ticker": "ELF", "Einstand": 109.00},
        {"Ticker": "ETSY", "Einstand": 67.00}, {"Ticker": "GTLB", "Einstand": 41.00}
    ])

col_tab, col_status = st.columns([1, 1.2])

with col_tab:
    with st.expander("Bestände editieren"):
        st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)

with col_status:
    for _, row in st.session_state.portfolio.iterrows():
        t, buyin = row['Ticker'], row['Einstand']
        curr_p = get_live_price(t)
        if curr_p:
            # Erweiterte Ampel-Logik für Repair
            diff_pct = (curr_p / buyin - 1) * 100
            if curr_p >= buyin:
                icon, stat = "🟢", "PROFIT: Call schreiben!"
            elif diff_pct > -20:
                icon, stat = "🟡", "REPAIR: Delta 0.10 wählen"
            else:
                icon, stat = "🔵", "DEEP REPAIR: Nur Delta 0.05!"
            
            st.write(f"{icon} **{t}**: {curr_p:.2f}$ ({diff_pct:.1f}%) → `{stat}`")

st.divider()

# --- REPAIR SCANNER ---
st.subheader("🔍 Repair Call Finder")
ticker = st.text_input("Ticker für Detail-Scan (z.B. HOOD)").strip().upper()

if ticker:
    price = get_live_price(ticker)
    p_row = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker]
    my_buyin = p_row.iloc[0]['Einstand'] if not p_row.empty else 0
    
    if price:
        st.info(f"Aktueller Kurs: {price:.2f}$ | Dein Einstand: {my_buyin:.2f}$")
        dates = get_all_expirations(ticker)
        if dates:
            sel_date = st.selectbox("Laufzeit (Repair ideal: 14-30 Tage)", dates)
            df = get_chain_for_date(ticker, sel_date, "call")
            
            if df is not None:
                # Wir filtern auf Strikes, die OTM sind
                df = df[df['strike'] > price].sort_values('strike')
                
                for _, row in df.head(15).iterrows():
                    d_abs = abs(float(row['delta']))
                    pop = (1 - d_abs) * 100
                    
                    # Logik für "Sicheren Repair"
                    # Wenn weit unter Einstand, ist Delta 0.10 das Maximum
                    is_repair_safe = d_abs <= 0.12
                    
                    if my_buyin > 0 and row['strike'] < my_buyin:
                        color = "🔵" if d_abs < 0.10 else "🟡"
                        note = "REPAIR MODE"
                    else:
                        color = "🟢" if d_abs < 0.15 else "🔴"
                        note = "TARGET REACHED"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {d_abs:.2f} | Chance: {pop:.0f}%"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Prämie", f"{row['mid']:.2f}$")
                        c2.metric("Abstand", f"{(row['strike']/price-1)*100:.1f}%")
                        
                        # Effektive Senkung des Einstands
                        new_basis = my_buyin - row['mid']
                        c3.metric("Basis neu", f"{new_basis:.2f}$", help="Dein neuer theoretischer Einstand")
                        
                        if d_abs > 0.15 and row['strike'] < my_buyin:
                            st.warning("⚠️ Achtung: Hohes Risiko einer Ausbuchung unter Einstand!")
                        else:
                            st.success("✅ Guter Repair-Kandidat: Geringes Risiko, senkt deine Kostenbasis.")
