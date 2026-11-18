"""
4_📈_Simulasyon_ve_CML: Monte Carlo Portföy Simülasyonu ve Sermaye Pazarı Doğrusu (CML)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.portfolio_engine import (
    calculate_returns,
    get_market_parameters,
    optimize_markowitz,
    simulate_monte_carlo,
)

st.set_page_config(page_title="Simülasyon & CML — Markowitzz", page_icon="📈", layout="wide")

st.title("📈 Monte Carlo Portföy Simülasyonu & Sermaye Pazarı Doğrusu (CML)")

prices = st.session_state.get("prices")

if prices is None or prices.empty:
    st.warning("⚠️ Lütfen önce **📥 1_Veri_Cekme** sayfasına giderek hisse senedi verilerini çekin.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Simülasyon Ayarları")

calc_mode = st.sidebar.radio(
    "🎯 Ölçek Modu:",
    ["Seçilen Dönem (Dönemsel)", "Yıllıklaştırılmış (252 Gün)"],
    index=0 if st.session_state.get("calculation_mode", "period") == "period" else 1,
)
mode_code = "period" if "Dönem" in calc_mode else "annualized"
st.session_state["calculation_mode"] = mode_code

annual_rf_pct = st.sidebar.number_input(
    "Yıllık Risksiz Faiz Oranı (%) [Örn: %45]:",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state.get("risk_free_rate_pct", 45.0)),
    step=1.0,
)
st.session_state["risk_free_rate_pct"] = annual_rf_pct
annual_rf = annual_rf_pct / 100.0

num_sims = st.sidebar.slider("Simüle Edilecek Portföy Sayısı:", 500, 5000, 2000, step=500)

# Parametreler
returns = calculate_returns(prices)
market_params = get_market_parameters(
    returns=returns,
    prices=prices,
    mode=mode_code,
    annual_trading_days=252,
    annual_risk_free_rate=annual_rf,
)

actual_days = market_params["actual_days"]
expected_returns = market_params["expected_returns"]
cov_matrix = market_params["covariance_matrix"]
scaled_rf = market_params["risk_free_rate"]
scale_title = f"{actual_days} İşlem Günü (Dönemsel)" if mode_code == "period" else "Yıllıklaştırılmış (252 Gün)"

# En iyi portföyleri hesapla
opt_results = optimize_markowitz(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=scaled_rf,
    max_weight=1.0,
)
max_sh = opt_results["max_sharpe"]
min_vol = opt_results["min_volatility"]

# Monte Carlo Simülasyonunu Çalıştır
with st.spinner(f"{num_sims} adet rastgele portföy simüle ediliyor..."):
    sim_df = simulate_monte_carlo(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=scaled_rf,
        max_weight=1.0,
        num_portfolios=num_sims,
    )

st.markdown(f"""
Bu sayfada, rastgele ağırlıklarla üretilen **{num_sims} farklı portföyün** getiri ve risk dağılımı gösterilmektedir.
* **Seçilen Ölçek:** {scale_title}
* **İndirgenmiş Risksiz Faiz ($R_f$):** %{scaled_rf * 100:.2f}
""")

# CML (Capital Market Line) Doğrusu Hesaplama
# Doğru: R = Rf + Sharpe * sigma
max_vol_axis = max(float(sim_df["Volatilite"].max()), max_sh["volatility"] * 1.5)
cml_vols = np.linspace(0, max_vol_axis, 50)
cml_returns = scaled_rf + max_sh["sharpe"] * cml_vols

fig_mc = go.Figure()

# 1. Simüle Edilen Rastgele Portföyler
fig_mc.add_trace(go.Scatter(
    x=sim_df["Volatilite"] * 100,
    y=sim_df["Beklenen Getiri"] * 100,
    mode="markers",
    marker=dict(
        color=sim_df["Sharpe"],
        colorscale="Viridis",
        size=4,
        opacity=0.6,
        colorbar=dict(title="Sharpe"),
        showscale=True,
    ),
    name="Rastgele Portföyler",
))

# 2. Sermaye Pazarı Doğrusu (CML)
fig_mc.add_trace(go.Scatter(
    x=cml_vols * 100,
    y=cml_returns * 100,
    mode="lines",
    line=dict(color="#F43F5E", width=2.5, dash="dash"),
    name="Sermaye Pazarı Doğrusu (CML)",
))

# 3. Risksiz Faiz Noktası (0, Rf)
fig_mc.add_trace(go.Scatter(
    x=[0],
    y=[scaled_rf * 100],
    mode="markers+text",
    text=["Rf (Risksiz Varlık)"],
    textposition="top right",
    name="Risksiz Faiz (Rf)",
    marker=dict(size=12, color="#F43F5E", symbol="triangle-up"),
))

# 4. Maksimum Sharpe Noktası (Teğet Noktası)
fig_mc.add_trace(go.Scatter(
    x=[max_sh["volatility"] * 100],
    y=[max_sh["return"] * 100],
    mode="markers+text",
    text=["⭐ Teğet Portföy (Max Sharpe)"],
    textposition="bottom right",
    name="Teğet Portföy",
    marker=dict(size=16, color="#10B981", symbol="star"),
))

# 5. Minimum Volatilite Noktası
fig_mc.add_trace(go.Scatter(
    x=[min_vol["volatility"] * 100],
    y=[min_vol["return"] * 100],
    mode="markers+text",
    text=["🛡️ Min Volatilite"],
    textposition="bottom left",
    name="Minimum Volatilite",
    marker=dict(size=14, color="#38BDF8", symbol="diamond"),
))

fig_mc.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (%) [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) [{scale_title}]",
    hovermode="closest",
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_mc, use_container_width=True)

# Simülasyon İstatistikleri
s1, s2, s3 = st.columns(3)
s1.metric("En Yüksek Simüle Sharpe", f"{sim_df['Sharpe'].max():.3f}")
s2.metric("En Düşük Simüle Volatilite", f"%{sim_df['Volatilite'].min() * 100:.2f}")
s3.metric("Ortalama Simüle Getiri", f"%{sim_df['Beklenen Getiri'].mean() * 100:.2f}")

st.markdown("""
> **💡 Finansal Not:** Sermaye Pazarı Doğrusu (CML), risksiz faiz oranı ($R_f$) ile Maksimum Sharpe (Teğet) portföyünü birleştiren en verimli sermaye tahsis doğrusudur. 
> Bu doğru üzerinde yatırımcı risk tercihine göre parasının bir kısmını faizde, bir kısmını da Teğet hisse sepetinde tutarak en yüksek getiri/risk oranına ulaşır.
""")
