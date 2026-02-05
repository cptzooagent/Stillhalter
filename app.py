import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP ---
st.set_page_config(page_title="Stillhalter Pro", layout="wide")

# Keys aus deinen Streamlit Secrets
MD_KEY = st.secrets["MARKETDATA_KEY"]
FINNHUB_KEY = st.secrets["FINNHUB_KEY"]

def get_clean_price(symbol):
    """Holt den Kurs via Finnhub (schnell & stabil)"""
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c'])
    except:
        return None

def get_marketdata_options(symbol):
    """Holt die komplette Put-Kette von MarketData"""
    # Wir laden die Kette für die nächsten verfügbaren Termine
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {
        "token": MD_KEY,
        "side": "put",
        "range": "otm" # Nur Out-of-the-money für Stillhalter
    }
    response = requests.get(url, params=params).json()
    
    if response.get('s') == 'ok':
        # MarketData liefert Daten in Listenform, die wir in ein DataFrame packen
        df = pd.DataFrame({
            'strike': response['strike'],
            'bid': response['bid'],
            'ask': response['ask'],
            'mid': response['mid'],
            'expiration': response['expiration'],
            'delta': response.get('delta', [0]*len(response['strike'])),
            'iv': response.get('iv', [0]*len(response['strike']))
        })
        return df
    return None

# --- APP INTERFACE ---
st.title("🛡️ Pro Stillhalter Scanner")

# Watchlist für schnellen Zugriff am Handy
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META"]
selected = st.pills("Favoriten", watchlist)
ticker = selected if selected else st.text_input("Ticker", "AAPL").upper()

if ticker:
    price = get_clean_price(ticker)
    
    if price:
        st.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        # Optionsdaten von MarketData laden
        with st.spinner('Lade Profi-Daten...'):
            chain = get_marketdata_options(ticker)
        
        if chain is not None:
            # Ablaufdaten sortieren
            expirations = sorted(chain['expiration'].unique())
            expiry = st.selectbox("Ablaufdatum wählen", expirations)
            
            # Filter auf gewähltes Datum
            df_expiry = chain[chain['expiration'] == expiry].copy()
            
            # Kennzahlen berechnen
            df_expiry['puffer'] = ((price / df_expiry['strike']) - 1) * 100
            
            st.subheader(f"Puts für {expiry}")
            
            # Darstellung der Top 5 Strikes (nach Nähe zum Kurs)
            for _, row in df_expiry.sort_values('strike', ascending=False).head(5).iterrows():
                # Rendite p.a. Schätzung
                dte = (datetime.strptime(row['expiration'], '%Y-%m-%d') - datetime.now()).days
                ann_return = (row['mid'] / row['strike']) * (365 / max(1, dte)) * 100
                
                with st.expander(f"Strike {row['strike']}$ | Delta: {abs(row['delta']):.2f} | {ann_return:.1f}% p.a."):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Preis (Mid):** {row['mid']}$")
                    c2.write(f"**Puffer:** {row['puffer']:.1f}%")
                    c3.write(f"**IV:** {row['iv']:.1%}")
        else:
            st.error("MarketData konnte keine Optionen finden. Limit erreicht?")
