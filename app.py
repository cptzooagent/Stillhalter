import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- SETUP & DESIGN ---
st.set_page_config(page_title="Stillhalter Pro Scanner", layout="wide")

# Keys aus den Streamlit Secrets
MD_KEY = st.secrets.get("MARKETDATA_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY")

# --- NEUE FUNKTIONEN FÜR WAHRSCHEINLICHKEIT ---

def get_sma_200(symbol):
    """Berechnet den SMA 200 (200-Tage-Linie) über Finnhub"""
    try:
        # Wir holen historische Tagesdaten der letzten 200+ Tage
        url = f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&count=250&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        if r.get('s') == 'ok':
            closes = r.get('c', [])
            return sum(closes[-200:]) / 200 if len(closes) >= 200 else None
    except:
        return None
    return None

def get_live_price(symbol):
    try:
        url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}'
        r = requests.get(url).json()
        return float(r['c']) if r.get('c') else None
    except:
        return None

def get_all_expirations(symbol):
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
    url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
    params = {"token": MD_KEY, "side": side, "expiration": date_str}
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

option_type = st.radio("Was möchtest du verkaufen?", ["Put 🛡️", "Call 📈"], horizontal=True)
side = "put" if "Put" in option_type else "call"

watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "META", "GOOGL", "AMZN"]
sel_fav = st.pills("Schnellauswahl", watchlist)
user_input = st.text_input("Ticker manuell", "")

ticker = (sel_fav if sel_fav else user_input).strip().upper()

if ticker:
    price = get_live_price(ticker)
    sma200 = get_sma_200(ticker)
    
    if price:
        # Spalten für Kurs und Trend-Indikator
        c_price, c_trend = st.columns(2)
        c_price.metric(f"Aktueller Kurs {ticker}", f"{price:.2f} $")
        
        if sma200:
            diff = ((price / sma200) - 1) * 100
            trend_icon = "✅ Bullish" if price > sma200 else "⚠️ Bearish"
            c_trend.metric("SMA 200 Check", f"{sma200:.2f} $", f"{diff:.1f}%")
            st.caption(f"Technischer Status: {trend_icon} (Kurs vs. 200-Tage-Linie)")

        all_dates = get_all_expirations(ticker)
        
        if all_dates:
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
                
                # OTM Filter
                if side == "put":
                    df_filtered = df[df['strike'] < price].sort_values('strike', ascending=False)
                else:
                    df_filtered = df[df['strike'] > price].sort_values('strike', ascending=True)
                
                for _, row in df_filtered.head(15).iterrows():
                    dte = max(1, (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days)
                    ann_return = (row['mid'] / (row['strike'] if side == "put" else price)) * (365 / dte) * 100
                    delta_val = abs(row['delta'])
                    puffer = (abs(price - row['strike']) / price) * 100
                    
                    # --- NEU: GEWINNWAHRSCHEINLICHKEIT (PoP) ---
                    # Statistische Annäherung: PoP = 1 - Delta
                    pop = (1 - delta_val) * 100
                    
                    # Ampel basierend auf Delta und Trend
                    color = "🟢" if delta_val < 0.16 else "🟡" if delta_val < 0.25 else "🔴"
                    
                    with st.expander(f"{color} Strike {row['strike']:.1f}$ | Chance: {pop:.0f}% | {ann_return:.1f}% p.a."):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Prämie", f"{row['mid']:.2f}$")
                        c2.metric("Puffer", f"{puffer:.1f}%")
                        c3.metric("Gewinn-Wahrsch.", f"{pop:.1f}%")
                        
                        st.progress(pop/100, help=f"Statistische Wahrscheinlichkeit von {pop:.1f}%, dass die Option wertlos verfällt.")
            else:
                st.warning(f"Keine {side}s gefunden.")
        else:
            st.error("Keine Laufzeiten verfügbar.")

st.divider()
st.link_button("Visualisieren in OptionStrat", f"https://optionstrat.com/visualizer/covered-call/{ticker}" if side == "call" else f"https://optionstrat.com/visualizer/cash-secured-put/{ticker}")
