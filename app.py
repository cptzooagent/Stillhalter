# =========================================================
# Stillhalter Depot Scanner v4
# POP • IV Rank • ROC • Earnings-Safe • Wheel-Ready
# GitHub + Streamlit Cloud ready
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
    page_title="🛡️ Stillhalter Depot Scanner v4",
    layout="wide"
)

# =========================================================
# GLOBAL PARAMETER
# =========================================================
RISK_FREE_RATE = 0.04
DEFAULT_IV = 0.40
EARNINGS_BLACKOUT_DAYS = 7

# =========================================================
# MATHE-FUNKTIONEN
# =========================================================
def calculate_bsm_delta(S, K, T, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

def probability_otm(S, K, T, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = (np.log(S / K) + (RISK_FREE_RATE - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d2) if option_type == "put" else 1 - norm.cdf(d2)

def calculate_rsi(close, window=14):
    if len(close) < window + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return float(100 - (100 / (1 + rs)).iloc[-1])

def calculate_iv_rank(iv_series):
    if iv_series.empty or iv_series.max() == iv_series.min():
        return 0.0
    return (iv_series.iloc[-1] - iv_series.min()) / (iv_series.max() - iv_series.min())

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

        hist = tk.history(period="3mo")
        rsi = calculate_rsi(hist["Close"]) if not hist.empty else 50

        # IV Rank basierend auf Volatilität der letzten 252 Tage
        iv_hist = tk.history(period="1y")["Close"].pct_change().rolling(20).std() * np.sqrt(252)
        iv_rank = calculate_iv_rank(iv_hist.dropna())

        earnings_date = None
        earnings_str = ""
        try:
            cal = tk.calendar
            if cal is not None and "Earnings Date" in cal:
                earnings_date = cal.loc["Earnings Date"][0]
                earnings_str = earnings_date.strftime("%d.%m.")
        except:
            pass

        return price, expiries, rsi, iv_rank, earnings_date, earnings_str

    except:
        return None, [], 50, 0.0, None, ""

# =========================================================
# SAFE OPTION CHAIN
# =========================================================
def safe_option_chain(ticker, expiry, option_type="puts"):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        return chain.puts if option_type == "puts" else chain.calls
    except:
        return pd.DataFrame()

# =========================================================
# PUT SCANNER
# =========================================================
def scan_puts(ticker, price, expiry, target_pop, min_roc, earnings_date):
    try:
        chain = safe_option_chain(ticker, expiry, "puts")
        if chain.empty:
            return None

        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        days = max(1, (expiry_dt - datetime.now()).days)
        T = days / 365

        # Earnings Filter
        if earnings_date and abs((earnings_date - expiry_dt).days) <= EARNINGS_BLACKOUT_DAYS:
            return None

        chain = chain[chain["bid"] > 0]

        strikes = chain["strike"].values
        iv = chain["impliedVolatility"].fillna(DEFAULT_IV).values

        pop = np.array([probability_otm(price, k, T, iv[i]) for i, k in enumerate(strikes)])
        chain["pop"] = pop
        chain = chain[chain["pop"] >= target_pop]

        if chain.empty:
            return None

        # ROC (Cash Secured Put)
        chain["roc"] = (chain["bid"] * 100) / (chain["strike"] * 100) * (365 / days) * 100
        chain = chain[chain["roc"] >= min_roc]

        if chain.empty:
            return None

        best = chain.sort_values("roc", ascending=False).iloc[0]

        return {
            "ticker": ticker,
            "strike": best["strike"],
            "bid": best["bid"],
            "roc": best["roc"],
            "pop": best["pop"],
            "buffer": abs(best["strike"] - price) / price * 100
        }
    except:
        return None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("🛡️ Stillhalter Regeln")

target_pop = st.sidebar.slider("Mindest-POP (%)", 70, 95, 85) / 100
min_roc = st.sidebar.slider("Mindest-ROC p.a. (%)", 10, 50, 20)
min_iv_rank = st.sidebar.slider("Min. IV Rank", 0.0, 1.0, 0.5)
max_tickers = st.sidebar.slider("Max. Ticker pro Scan", 10, 60, 40)

# =========================================================
# UI – SCANNER
# =========================================================
st.title("🛡️ Stillhalter Depot Scanner v4")

if st.button("🚀 Scan starten"):
    results = []
    watchlist = get_watchlist()[:max_tickers]

    prog = st.progress(0.0)
    status = st.empty()

    for i, ticker in enumerate(watchlist):
        if i % 5 == 0:
            status.text(f"Analysiere {ticker}...")

        price, expiries, rsi, iv_rank, earn_dt, earn_str = load_stock_data(ticker)

        if not price or not expiries or iv_rank < min_iv_rank:
            continue

        expiry = min(expiries, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - datetime.now()).days - 30))

        trade = scan_puts(ticker, price, expiry, target_pop, min_roc, earn_dt)

        if trade:
            trade.update({"rsi": rsi, "iv_rank": iv_rank, "earn": earn_str})
            results.append(trade)

        prog.progress((i + 1) / len(watchlist))

    status.text("Scan abgeschlossen ✔")

    if not results:
        st.warning("Keine geeigneten Trades gefunden.")
    else:
        df = pd.DataFrame(results).sort_values("roc", ascending=False).head(12)
        cols = st.columns(4)

        for i, row in enumerate(df.to_dict("records")):
            with cols[i % 4]:
                st.markdown(f"### {row['ticker']}")
                st.metric("ROC p.a.", f"{row['roc']:.1f}%")
                st.write(f"🎯 Strike: **{row['strike']:.1f}$**")
                st.write(f"💰 Bid: **{row['bid']:.2f}$**")
                st.write(f"🧮 POP: **{row['pop']*100:.1f}%**")
                st.write(f"🛡️ Puffer: **{row['buffer']:.1f}%**")
                st.caption(f"RSI: {row['rsi']:.0f} | IV Rank: {row['iv_rank']:.2f}")

                # Wheel-Kandidaten markieren
                if row["rsi"] < 45:
                    st.success("🔁 Wheel-Kandidat")

                # Earnings Warnung
                if row["earn"]:
                    st.warning(f"📅 Earnings: {row['earn']}")
