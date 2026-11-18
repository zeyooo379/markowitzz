"""
Hisse senedi verisi çekme ve önbelleğe alma modülü
"""

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def hisse_verisi_cek(semboller: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Belirtilen BIST sembolleri için kapanış fiyatlarını çeker.
    Sembollerin sonuna otomatik olarak '.IS' eklenir.
    """
    if not semboller:
        return pd.DataFrame()

    yf_tickers = [f"{s}.IS" if not s.endswith(".IS") else s for s in semboller]
    
    try:
        data = yf.download(
            tickers=yf_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame()

        if "Adj Close" in data.columns:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns:
            prices = data["Close"].copy()
        else:
            return pd.DataFrame()

        # Tek hisse çekildiğinde Series yerine DataFrame yap
        if isinstance(prices, pd.Series):
            col_name = semboller[0].replace(".IS", "")
            prices = prices.to_frame(name=col_name)
        else:
            prices.columns = [col.replace(".IS", "") for col in prices.columns]

        prices = prices.dropna(how="all").ffill().bfill()
        return prices

    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def endeks_verisi_cek(period: str = "6mo") -> pd.DataFrame:
    """
    BIST 100 endeksi verilerini çeker.
    """
    try:
        df = yf.download("XU100.IS", period=period, progress=False)
        if not df.empty and "Close" in df.columns:
            close_series = df["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            return pd.DataFrame({"XU100": close_series}).dropna()
    except Exception:
        pass
    return pd.DataFrame()
