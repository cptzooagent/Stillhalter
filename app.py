import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP & DESIGN ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- FUNKTIONEN ---

def get_live_price(symbol):
    """Holt den aktuellen Kurs über Finnhub"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if r.get('c') else None
    except:
        return None

def get_all_expirations(symbol):
    """Holt JEDES verfügbare Verfallsdatum für den Ticker"""
    url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/"
    params = {"token": MD_KEY}
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            return sorted(response.get('expirations', []))
    except:
        return []
    return []

def get_chain_for_date(symbol, date_str, side):
    """Holt die Strikes für ein Datum und die gewählte Seite (call/put)"""
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {
        "token": MD_KEY,
        "side": side,
        "expiration": date_str
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            return pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'delta': response.get('delta', [0] * len(response['strike'])),
                'iv': response.get('iv', [0] * len(response['strike'])),
                'expiration': date_str
            })
    except:
        return None
    return None

def format_date_label(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        dte = (dt - datetime.now()).days
        return f"{dt.strftime('%d.%m.%Y')} ({max(0, dte)} Tage)"
    except:
        return date_str

# --- USER INTERFACE ---
st.title("🛡️ Pro Stillhalter Scanner")

# NEU: Wahl zwischen Put und Call
option_type = st.radio("Was möchtest du verkaufen?", ["Put 🛡️", "Call 📈"], horizontal=True)
side = "put" if "Put" in option_type else "call"

# Favoriten-Leiste
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Schnellauswahl", watchlist)
user_input = st.text_input("Ticker manuell", "")

ticker = (sel_fav if sel_fav else user_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        all_dates = get_all_expirations(ticker)
        
        if all_dates:
            # Vorauswahl-Logik (30-50 Tage)
            default_index = 0
            for i, d in enumerate(all_dates):
                days = (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days
                if 30 <= days <= 50:
                    default_index = i
                    break

            selected_date = st.selectbox("Laufzeit wählen", all_dates, index=default_index, format_func=format_date_label)
            
            with st.spinner(f"Lade {side}s..."):
                df = get_chain_for_date(ticker, selected_date, side)
            
            if df is not None and not df.empty:
                st.subheader(f"{'Puts' if side == 'put' else 'Calls'} für {format_date_label(selected_date)}")
                
                # Filter-Logik: Puts < Kurs | Calls > Kurs (OTM)
                if side == "put":
                    df_filtered = df[df['strike'] < price].sort_values('strike', ascending=False)
                else:
                    df_filtered = df[df['strike'] > price].sort_values('strike', ascending=True)
                
                for _, row in df_filtered.head(15).iterrows():
                    dte = max(1, (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days)
                    ann_return = (row['mid'] / (row['strike'] if side == "put" else price)) * (365 / dte) * 100
                    delta_val = abs(row['delta'])
                    puffer = (abs(price - row['strike']) / price) * 100
                    
                    # Ampel: Grün bis 0.16 Delta, Gelb bis 0.25, danach Rot
                    color = "🟢" if delta_val < 0.16 else "🟡" if delta_val < 0.25 else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Prämie", f"{row['mid']:.2f}$")
                        c2.metric("Abstand", f"{puffer:.1f}%")
                        c3.metric("Rendite p.a.", f"{ann_return:.1f}%")
            else:
                st.warning(f"Keine {side}s gefunden.")
        else:
            st.error("Keine Laufzeiten verfügbar.")

st.divider()
st.link_button("Visualisieren in OptionStrat", f"https://optionstrat.com/visualizer/covered-call/{ticker}" if side == "call" else f"https://optionstrat.com/visualizer/cash-secured-put/{ticker}")
