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
        if r.get('c'):
            return float(r['c'])
    except:
        return None
    return None

def get_marketdata_options(symbol):
    """Holt die gesamte Optionskette (alle Laufzeiten) von MarketData.app"""
    # Wir lassen 'range' weg, um alle verfügbaren Verfallstermine zu erhalten
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {
        "token": MD_KEY, 
        "side": "put"
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('s') == 'ok':
            df = pd.DataFrame({
                'strike': response['strike'],
                'mid': response['mid'],
                'expiration': response['expiration'],
                'delta': response.get('delta', [0] * len(response['strike'])),
                'iv': response.get('iv', [0] * len(response['strike']))
            })
            
            # WICHTIG: Sofortige Umwandlung der Unix-Zahlen in Text-Datum (YYYY-MM-DD)
            def fix_date(x):
                try:
                    if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()):
                        return datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d')
                    return str(x)
                except:
                    return str(x)
            
            df['expiration'] = df['expiration'].apply(fix_date)
            return df
    except:
        return None
    return None

def format_expiry_with_dte(date_str):
    """Formatierungsfunktion für das Dropdown-Menü (Datum + DTE)"""
    try:
        exp_dt = datetime.strptime(date_str, '%Y-%m-%d')
        days_to_expiry = (exp_dt - datetime.now()).days
        return f"{exp_dt.strftime('%d.%m.%Y')} ({max
