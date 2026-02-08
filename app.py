# =========================================================
# CapTrader Stillhalter Depot – Scanner v2
# Clean, robust, depot-orientiert
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from scipy.stats import norm
from datetime import datetime

# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="🛡️ Stillhalter Depot Scanner v2",
    layout="wide"
)

# =========================================================
# GLOBAL SETTINGS
# =========================================================

RISK_FREE_RATE = 0.04
DEFAULT_IV = 0.40
EARNINGS_BLACKOUT_DAYS = 5

# =========================================================
# MATHE & INDIKATOREN
# =========================================================

def calculate_bsm_delta(S, K, T, sigma, r=RISK_FREE_RATE, option_type="put"):
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def calculate_rsi(close_prices, window=14):
    if len(close_prices) < window + 1:
        return 50.0

    delta = close_prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss

    return float(100 - (100 / (1 + rs)).iloc[-1])


# =========================================================
# WATCHLIST
# =========================================================

@st.cache_data(ttl=3600)
def get_watchlist():
    return sorted(list(set([
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ADBE","NFLX",
        "AMD","INTC","QCOM","AMAT","TXN","MU","ISRG","LRCX","PANW","SNPS",
        "LLY","V","MA","JPM","WMT","XOM","UNH","PG","ORCL","COST",
        "ABBV","BAC","KO","PEP","CRM","WFC","DIS","CAT","AXP","IBM",
        "COIN","MARA","PLTR","AFRM","SQ","RIVN","UPST","HOOD","SOFI","MSTR"
    ])))


# =========================================================
# MARKTDATEN
# =========================================================

@st.cache_data(ttl=900)
def load_stock_data(ticker):
    try:
        tk = yf.Ticker(ticker)

        price = tk.fast_info.get("last_price")
        expiries = list(tk.options)

        hist = tk.history(period="1mo")
        rsi = calculate_rsi(hist["Close"]) if not hist.empty else 50

        earnings_date = None
        earnings_str = ""

        try:
            cal = tk.calendar
            if cal is not None and "Earnings Date" in cal:
                earnings_date = cal.loc["Earnings Date"][0]
                earnings_str = earnings_date.strftime("%d.%m.")
        except:
            pass

        return price, expiries, rsi, earnings_date, earnings_str

    except:
        return None, [], 50, None, ""


# =========================================================
# OPTION ANALYSE
# =========================================================

def find_best_put(
    ticker,
    price,
    expiry,
    max_delta,
    min_yield_pa,
    earnings_date
):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry).puts

        if chain.empty:
            return None

        days = max(1, (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days)
        T = days / 365

        # Earnings Filter
        if earnings_date:
            if abs((earnings_date - datetime.strptime(expiry, "%Y-%m-%d")).days) <= EARNINGS_BLACKOUT_DAYS:
                return None

        iv = chain["impliedVolatility"].fillna(DEFAULT_IV).values
        strikes = chain["strike"].values

        deltas = np.array([
            calculate_bsm_delta(price, k, T, iv[i])
            for i, k in enumerate(strikes)
        ])

        chain["delta"] = np.abs(deltas)
        chain = chain[chain["delta"] <= max_delta]

        if chain.empty:
            return None

        chain["yield_pa"] = (chain["bid"] / chain["strike"]) * (365 / days) * 100
        chain = chain[chain["yield_pa"] >= min_yield_pa]

        if chain.empty:
            return None

        best = chain.sort_values("yield_pa", ascending=False).iloc[0]

        return {
            "ticker": ticker,
            "strike": best["strike"],
            "bid": best["bid"],
            "delta": best["delta"],
            "yield": best["yield_pa"],
            "buffer": abs(best["strike"] - price) / price * 100
        }

    except:
        return None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🛡️ Stillhalter Parameter")

target_prob = st.sidebar.slider("Sicherheit (OTM %)", 70, 95, 83)
max_delta = (100 - target_prob) / 100

min_yield_pa = st.sidebar.number_input(
    "Mindestrendite p.a. (%)",
    value=20,
    step=5
)

# =========================================================
# UI – SCANNER
# =========================================================

st.title("🛡️ Stillhalter Depot Scanner v2")

if st.button("🚀 Scan starten"):
    watchlist = get_watchlist()
    results = []

    prog = st.progress(0.0)
    status = st.empty()

    for i, ticker in enumerate(watchlist):
        status.text(f"Analysiere {ticker}...")
        prog.progress((i + 1) / len(watchlist))

        price, expiries, rsi, earn_dt, earn_str = load_stock_data(ticker)

        if not price or not expiries:
            continue

        expiry = min(
            expiries,
            key=lambda x: abs(
                (datetime.strptime(x, "%Y-%m-%d") - datetime.now()).days - 30
            )
        )

        trade = find_best_put(
            ticker,
            price,
            expiry,
            max_delta,
            min_yield_pa,
            earn_dt
        )

        if trade:
            trade["rsi"] = rsi
            trade["earn"] = earn_str
            results.append(trade)

    status.text("Scan abgeschlossen ✔")

    if not results:
        st.warning("Keine geeigneten Stillhalter-Trades gefunden.")
    else:
        df = pd.DataFrame(results).sort_values("yield", ascending=False).head(12)

        cols = st.columns(4)
        for i, row in enumerate(df.to_dict("records")):
            with cols[i % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("Rendite p.a.", f"{row['yield']:.1f}%")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$**")
                st.write(f"💰 Bid: **{row['bid']:.2f}$**")
                st.write(f"📉 Delta: **{row['delta']:.2f}**")
                st.write(f"🛡️ Puffer: **{row['buffer']:.1f}%**")
                st.caption(f"RSI: {row['rsi']:.0f}")
                if row["earn"]:
                    st.warning(f"⚠️ Earnings: {row['earn']}")

