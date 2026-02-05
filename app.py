import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- FUNKTIONEN ---

def get_live_price(symbol):
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if r.get('c') else None
    except: return None

def get_all_expirations(symbol):
    """Holt JEDES verfügbare Verfallsdatum für den Ticker"""
    url = f"https://api.marketdata.app/v1/options/expirations/{symbol}/"
    params = {"token": MD_KEY}
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            # MarketData liefert hier oft direkt eine Liste von Daten
            return sorted(response.get('expirations', []))
    except: return []
    return []

def get_chain_for_date(symbol, date_str):
    """Holt die Strikes für ein ganz spezifisches Datum"""
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {
        "token": MD_KEY,
        "side": "put",
        "expiration": date_str # Hier erzwingen wir das gewählte Datum
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            return pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'delta': response.get('delta', [0] * len(response['strike'])),
                'expiration': date_str
            })
    except: return None
    return None

def format_date_label(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        dte = (dt - datetime.now()).days
        return f"{dt.strftime('%d.%m.%Y')} ({max(0, dte)} Tage)"
    except: return date_str

# --- INTERFACE ---
st.title("🛡️ Pro Stillhalter Scanner")

watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Schnellauswahl", watchlist)
ticker = (sel_fav if sel_fav else st.text_input("Ticker", "AAPL")).strip().upper()

if ticker:
    price = get_live_price(ticker)
    if price:
        st.metric(f"Kurs {ticker}", f"{price:.2f} $")
        
        # SCHRITT 1: Alle Termine laden
        all_dates = get_all_expirations(ticker)
        
        if all_dates:
            # SCHRITT 2: Datum auswählen lassen
            selected_date = st.selectbox(
                "Wähle eine Laufzeit", 
                all_dates, 
                format_func=format_date_label,
                index=min(2, len(all_dates)-1) # Vorauswahl auf ca. 30-45 Tage
            )
            
            # SCHRITT 3: Nur die Daten für dieses Datum laden
            with st.spinner("Lade Strikes..."):
                df = get_chain_for_date(ticker, selected_date)
            
            if df is not None and not df.empty:
                st.subheader(f"Puts für den {format_date_label(selected_expiry if 'selected_expiry' in locals() else selected_date)}")
                
                # Nur OTM Strikes
                df = df[df['strike'] < price].sort_values('strike', ascending=False).head(15)
                
                for _, row in df.iterrows():
                    dte = max(1, (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days)
                    ann_return = (row['mid'] / row['strike']) * (365 / dte) * 100
                    delta_val = abs(row['delta'])
                    puffer = ((price / row['strike']) - 1) * 100
                    
                    color = "🟢" if delta_val < 0.16 else "🟡" if delta_val < 0.25 else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Delta: {delta_val:.2f} | {ann_return:.1f}% p.a."):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Prämie", f"{row['mid']:.2f}$")
                        c2.metric("Puffer", f"{puffer:.1f}%")
                        c3.metric("Rendite", f"{ann_return:.1f}%")
            else:
                st.warning("Keine Strikes für dieses Datum gefunden.")
        else:
            st.error("Keine Laufzeiten gefunden. API-Key oder Ticker prüfen.")

st.divider()
st.link_button("In OptionStrat öffnen", f"https://optionstrat.com/visualizer/cash-secured-put/{ticker}")
