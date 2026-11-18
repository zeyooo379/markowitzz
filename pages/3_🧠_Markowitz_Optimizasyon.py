"""
3_🧠_Markowitz_Optimizasyon: Portföy Ağırlıkları ve Etkin Sınır (Efficient Frontier)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.constants import BIST_HISSELER, RENKLER
from utils.portfolio_engine import (
    calculate_returns,
    generate_efficient_frontier,
    get_market_parameters,
    optimize_markowitz,
)

st.set_page_config(page_title="Markowitz Optimizasyonu — Markowitzz", page_icon="🧠", layout="wide")

st.title("🧠 Markowitz Portföy Optimizasyonu & Etkin Sınır")

prices = st.session_state.get("prices")

if prices is None or prices.empty:
    st.warning("⚠️ Lütfen önce **📥 1_Veri_Cekme** sayfasına giderek hisse senedi verilerini çekin.")
    st.stop()

# --- Sidebar Ayarları ---
st.sidebar.header("⚙️ Optimizasyon Ayarları")

calc_mode = st.sidebar.radio(
    "🎯 Optimizasyon Ölçeği:",
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

max_weight_pct = st.sidebar.slider(
    "Tek Hisseye İzin Verilen Max Ağırlık (%):",
    min_value=int(100 / len(prices.columns)),
    max_value=100,
    value=100,
    step=5,
    help="Örneğin %40 seçerseniz, hiçbir hisseye portföyün %40'ından fazlası verilmez.",
)
max_weight = max_weight_pct / 100.0

# Parametreleri Hesapla
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

# Optimizasyonu Çalıştır
opt_results = optimize_markowitz(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=scaled_rf,
    max_weight=max_weight,
)

min_vol = opt_results["min_volatility"]
max_sh = opt_results["max_sharpe"]
eq_w = opt_results["equal_weight"]

# Bilgi Çubuğu
st.success(f"""
✅ **Optimizasyon Tamamlandı!**  
* **Ölçek:** {scale_title}  
* **Kullanılan Dönemsel Risksiz Faiz:** %{scaled_rf * 100:.2f} *(Yıllık %{annual_rf_pct:.1f} baz alınarak hesaplandı)*  
* **Hisse Sayısı:** {len(prices.columns)} hisse | **Max Ağırlık Sınırı:** %{max_weight_pct}
""")

st.markdown("---")

# =============================================================================
# 1. OPTİMAL PORTFÖY ÖZET METRİKLERİ
# =============================================================================
st.subheader("🏆 Portföy Karşılaştırma Özeti")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🥇 Maksimum Sharpe Portföyü")
    st.caption("Risk başına en yüksek getiriyi sunan Teğet (Tanjant) Portföy.")
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{max_sh['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{max_sh['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{max_sh['sharpe']:.3f}")

with c2:
    st.markdown("### 🛡️ Minimum Volatilite Portföyü")
    st.caption("Piyasadaki en düşük riskli ve en defansif portföy dağılımı.")
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{min_vol['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{min_vol['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{min_vol['sharpe']:.3f}")

with c3:
    st.markdown("### ⚖️ Eşit Ağırlıklı Portföy (1/N)")
    st.caption("Her hisseye eşit pay (%{:.1f}) verilen karşılaştırma portföyü.".format(100 / len(prices.columns)))
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{eq_w['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{eq_w['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{eq_w['sharpe']:.3f}")

st.markdown("---")

# =============================================================================
# 2. AĞIRLIK DAĞILIMLARI (GRAFİK & TABLO)
# =============================================================================
st.subheader("🥧 Optimal Portföy Ağırlıkları")

col_left, col_right = st.columns(2)

weights_df = pd.DataFrame({
    "Sembol": prices.columns,
    "Şirket": [BIST_HISSELER.get(s, s) for s in prices.columns],
    "Max Sharpe Ağırlık (%)": (max_sh["weights"] * 100).round(2).values,
    "Min Volatilite Ağırlık (%)": (min_vol["weights"] * 100).round(2).values,
    "Eşit Ağırlık (%)": (eq_w["weights"] * 100).round(2).values,
})

with col_left:
    st.markdown("##### 🥇 Maksimum Sharpe Dağılımı")
    fig_pie_sh = px.pie(
        weights_df[weights_df["Max Sharpe Ağırlık (%)"] > 0.01],
        names="Sembol",
        values="Max Sharpe Ağırlık (%)",
        hole=0.45,
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_pie_sh.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie_sh.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie_sh, use_container_width=True)

with col_right:
    st.markdown("##### 🛡️ Minimum Volatilite Dağılımı")
    fig_pie_vol = px.pie(
        weights_df[weights_df["Min Volatilite Ağırlık (%)"] > 0.01],
        names="Sembol",
        values="Min Volatilite Ağırlık (%)",
        hole=0.45,
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    fig_pie_vol.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie_vol.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie_vol, use_container_width=True)

st.dataframe(weights_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# 3. ETKİN SINIR (EFFICIENT FRONTIER) GRAFİĞİ
# =============================================================================
st.subheader("📈 Etkin Sınır (Efficient Frontier) Eğrisi")
st.caption(f"Yatay eksen: Volatilite (Risk), Dikey eksen: Beklenen Getiri ({scale_title}).")

with st.spinner("Etkin Sınır eğrisi hesaplanıyor..."):
    frontier_df = generate_efficient_frontier(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=scaled_rf,
        max_weight=max_weight,
        points=40,
    )

fig_ef = go.Figure()

# 1. Etkin Sınır Eğrisi
if not frontier_df.empty:
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["Volatilite"] * 100,
        y=frontier_df["Beklenen Getiri"] * 100,
        mode="lines",
        name="Etkin Sınır (Efficient Frontier)",
        line=dict(color="#38BDF8", width=3),
    ))

# 2. Bireysel Hisseler
single_vols = market_params["volatilities"] * 100
single_rets = expected_returns * 100

fig_ef.add_trace(go.Scatter(
    x=single_vols.values,
    y=single_rets.values,
    mode="markers+text",
    text=single_vols.index,
    textposition="top center",
    name="Tekil Hisseler",
    marker=dict(size=10, color="#F59E0B", symbol="circle"),
))

# 3. Maksimum Sharpe Noktası
fig_ef.add_trace(go.Scatter(
    x=[max_sh["volatility"] * 100],
    y=[max_sh["return"] * 100],
    mode="markers+text",
    text=["⭐ Max Sharpe"],
    textposition="bottom right",
    name="Maksimum Sharpe",
    marker=dict(size=16, color="#10B981", symbol="star"),
))

# 4. Minimum Volatilite Noktası
fig_ef.add_trace(go.Scatter(
    x=[min_vol["volatility"] * 100],
    y=[min_vol["return"] * 100],
    mode="markers+text",
    text=["🛡️ Min Volatilite"],
    textposition="top right",
    name="Minimum Volatilite",
    marker=dict(size=14, color="#EF4444", symbol="diamond"),
))

# 5. Eşit Ağırlıklı Nokta
fig_ef.add_trace(go.Scatter(
    x=[eq_w["volatility"] * 100],
    y=[eq_w["return"] * 100],
    mode="markers+text",
    text=["⚖️ Eşit Ağırlık (1/N)"],
    textposition="bottom left",
    name="Eşit Ağırlıklı",
    marker=dict(size=12, color="#A855F7", symbol="square"),
))

fig_ef.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (%) [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) [{scale_title}]",
    hovermode="closest",
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_ef, use_container_width=True)

st.info("👉 Sol menüden **📈 4_Simulasyon_ve_CML** sayfasına giderek 2000+ portföylük Monte Carlo simülasyonunu ve Sermaye Pazarı Doğrusunu (CML) inceleyebilirsiniz.")
