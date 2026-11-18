"""
3_🧠_Markowitz_Optimizasyon: Portföy Ağırlıkları, Etkin Sınır (Efficient Frontier) ve İnteraktif Risk Keşfi
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

st.set_page_config(
    page_title="Markowitz Optimizasyonu & Etkin Sınır — Markowitzz",
    page_icon="🧠",
    layout="wide",
)

# Özel CSS Tasarımı
st.markdown("""
<style>
    .guide-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #38BDF8;
        margin-bottom: 15px;
    }
    .guide-card-green {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #10B981;
        margin-bottom: 15px;
    }
    .guide-card-orange {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #F59E0B;
        margin-bottom: 15px;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Markowitz Portföy Optimizasyonu & Etkin Sınır")
st.markdown("""
Markowitz Modern Portföy Teorisi; hisselerinizi öyle bir oranla dağıtmanızı sağlar ki, 
**aynı risk seviyesinde mümkün olan en yüksek getiriyi** elde edersiniz.
""")

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
    help="Örneğin %40 seçilirse hiçbir hisseye portföyün %40'ından fazlası verilmez.",
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

# =============================================================================
# 1. TEMEL PORTFÖYLER ÖZETİ
# =============================================================================
st.subheader("🏆 Temel Portföy Stratejileri Karşılaştırması")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="guide-card-green">
        <h4 style="color: #10B981; margin-bottom: 5px;">🥇 Maksimum Sharpe Portföyü</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">Risk başına en yüksek getiriyi sunan en akılcı ve dengeli portföy.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{max_sh['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{max_sh['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{max_sh['sharpe']:.3f}")

with c2:
    st.markdown("""
    <div class="guide-card">
        <h4 style="color: #38BDF8; margin-bottom: 5px;">🛡️ Minimum Volatilite Portföyü</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">"Geceleri rahat uyumak istiyorum" diyenler için dalgalanması en düşük sepet.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{min_vol['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{min_vol['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{min_vol['sharpe']:.3f}")

with c3:
    st.markdown("""
    <div class="guide-card-orange">
        <h4 style="color: #F59E0B; margin-bottom: 5px;">⚖️ Eşit Ağırlıklı Portföy (1/N)</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">"Hiç analiz yapmayıp her hisseye eşit pay verseydim ne olurdu?" kıyası.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{eq_w['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{eq_w['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{eq_w['sharpe']:.3f}")

st.markdown("---")

# =============================================================================
# 2. AĞIRLIK DAĞILIMLARI (GRAFİK & TABLO)
# =============================================================================
st.subheader("🥧 Optimal Portföy Dağılımları (Hangi Hisseden Ne Kadar Almalı?)")

col_left, col_right = st.columns(2)

weights_df = pd.DataFrame({
    "Sembol": prices.columns,
    "Şirket": [BIST_HISSELER.get(s, s) for s in prices.columns],
    "Max Sharpe Ağırlık (%)": (max_sh["weights"] * 100).round(2).values,
    "Min Volatilite Ağırlık (%)": (min_vol["weights"] * 100).round(2).values,
    "Eşit Ağırlık (%)": (eq_w["weights"] * 100).round(2).values,
})

with col_left:
    st.markdown("##### 🥇 Maksimum Sharpe Portföyü Dağılımı")
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
    st.markdown("##### 🛡️ Minimum Volatilite Portföyü Dağılımı")
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
# 3. ETKİN SINIR (EFFICIENT FRONTIER) KILAVUZU & GRAFİĞİ
# =============================================================================
st.subheader("📈 Etkin Sınır (Efficient Frontier) Nasıl Anlaşılır?")

# 3 Adımda Anlaşılır Kılavuz
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    <div class="guide-card">
        <h5 style="color: #38BDF8;">🧭 1. Altın Kural: Kuzeybatı</h5>
        <p style="color: #CBD5E1; font-size: 0.9rem;">
            Bir yatırımcı daima <b>daha yüksek getiri (yukarı)</b> ve <b>daha düşük risk (sola)</b> ister. 
            Bu yüzden grafikte sol-üst köşe (Kuzeybatı) en ideal yerdir.
        </p>
    </div>
    """, unsafe_allow_html=True)

with g2:
    st.markdown("""
    <div class="guide-card-green">
        <h5 style="color: #10B981;">🚀 2. Mavi Eğri: Ulaşılabilecek Zirve</h5>
        <p style="color: #CBD5E1; font-size: 0.9rem;">
            Mavi çizgi, elinizdeki hisselerle oluşturabileceğiniz <b>en mükemmel portföylerin sınırıdır</b>. 
            Bu eğri üzerindeki her nokta, o risk seviyesi için <i>mümkün olan en yüksek getiriyi</i> verir.
        </p>
    </div>
    """, unsafe_allow_html=True)

with g3:
    st.markdown("""
    <div class="guide-card-orange">
        <h5 style="color: #F59E0B;">⚠️ 3. Eğrinin Altı: Verimsiz Bölge</h5>
        <p style="color: #CBD5E1; font-size: 0.9rem;">
            Tekil hisseler (turuncu noktalar) genelde eğrinin altında kalır. 
            Bunun anlamı: <i>"Aynı riski alıp daha az kazanıyorsunuz; sepet yapsaydınız aynı riske daha çok kazanırdınız!"</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Etkin Sınır Verisi Hesapla
with st.spinner("Etkin Sınır eğrisi ve optimal noktalar hesaplanıyor..."):
    frontier_df = generate_efficient_frontier(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=scaled_rf,
        max_weight=max_weight,
        points=50,
    )

fig_ef = go.Figure()

# 1. Etkin Sınır Eğrisi
if not frontier_df.empty:
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["Volatilite"] * 100,
        y=frontier_df["Beklenen Getiri"] * 100,
        mode="lines",
        name="Etkin Sınır (En Verimli Portföyler)",
        line=dict(color="#38BDF8", width=3.5),
        customdata=frontier_df["Sharpe"],
        hovertemplate="<b>Etkin Sınır Portföyü</b><br>Risk (Volatilite): %{x:.2f}%<br>Beklenen Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
    ))

# 2. Tekil Hisseler
single_vols = market_params["volatilities"] * 100
single_rets = expected_returns * 100
single_sharpes = (single_rets - scaled_rf * 100) / np.maximum(single_vols, 1e-6)

fig_ef.add_trace(go.Scatter(
    x=single_vols.values,
    y=single_rets.values,
    mode="markers+text",
    text=single_vols.index,
    textposition="top center",
    name="Tekil Hisseler",
    marker=dict(size=11, color="#F59E0B", symbol="circle", line=dict(color="#FFFFFF", width=1)),
    customdata=single_sharpes.values,
    hovertemplate="<b>Hisse: %{text}</b><br>Risk (Volatilite): %{x:.2f}%<br>Beklenen Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
))

# 3. Maksimum Sharpe Noktası (Yıldız)
fig_ef.add_trace(go.Scatter(
    x=[max_sh["volatility"] * 100],
    y=[max_sh["return"] * 100],
    mode="markers+text",
    text=["⭐ En Verimli Portföy (Max Sharpe)"],
    textposition="bottom right",
    name="Max Sharpe",
    marker=dict(size=18, color="#10B981", symbol="star", line=dict(color="#FFFFFF", width=1.5)),
    hovertemplate="<b>Maksimum Sharpe Portföyü</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: " + f"{max_sh['sharpe']:.2f}<extra></extra>",
))

# 4. Minimum Volatilite Noktası (Elmas)
fig_ef.add_trace(go.Scatter(
    x=[min_vol["volatility"] * 100],
    y=[min_vol["return"] * 100],
    mode="markers+text",
    text=["🛡️ En Güvenli Portföy (Min Volatilite)"],
    textposition="top left",
    name="Min Volatilite",
    marker=dict(size=15, color="#EF4444", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
    hovertemplate="<b>Minimum Volatilite Portföyü</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: " + f"{min_vol['sharpe']:.2f}<extra></extra>",
))

# 5. Eşit Ağırlıklı Nokta (Kare)
fig_ef.add_trace(go.Scatter(
    x=[eq_w["volatility"] * 100],
    y=[eq_w["return"] * 100],
    mode="markers+text",
    text=["⚖️ Eşit Dağılım (1/N)"],
    textposition="bottom left",
    name="Eşit Ağırlıklı",
    marker=dict(size=13, color="#A855F7", symbol="square", line=dict(color="#FFFFFF", width=1)),
    hovertemplate="<b>Eşit Ağırlıklı Portföy</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<extra></extra>",
))

# Grafiğe Açıklayıcı Ok ve Etiket Ekleyelim
fig_ef.add_annotation(
    x=max_sh["volatility"] * 100,
    y=max_sh["return"] * 100,
    text="Risk başına en çok getiri burada!",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowcolor="#10B981",
    ax=60,
    ay=-40,
    bgcolor="#1E293B",
    bordercolor="#10B981",
    font=dict(color="#F8FAFC", size=11),
)

fig_ef.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (Standart Sapma) (%) — [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) — [{scale_title}]",
    hovermode="closest",
    height=550,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_ef, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. 🎯 İNTERAKTİF RİSK SEVİYESİ SEÇİCİ (CUSTOM RISK EXPLORER)
# =============================================================================
st.subheader("🎯 Kendi Risk Seviyenizi Seçin: Size Özel Portföyü Görün")
st.markdown("""
Aşağıdaki kaydırma çubuğundan (slider) **kendi kabul edebileceğiniz risk seviyesini** belirleyin. 
Sistem, Etkin Sınır üzerindeki en ideal noktayı bulup **hangi hisseden yüzde kaç almanız gerektiğini** hemen listeler!
""")

if not frontier_df.empty:
    min_available_vol = float(frontier_df["Volatilite"].min() * 100)
    max_available_vol = float(frontier_df["Volatilite"].max() * 100)
    
    # Slider
    user_target_vol = st.slider(
        f"Almak İstediğiniz Risk Seviyesi (%) [{scale_title}]:",
        min_value=round(min_available_vol, 2),
        max_value=round(max_available_vol, 2),
        value=round(max_sh["volatility"] * 100, 2),
        step=0.25,
        help="Riski artırdıkça beklenen getiri yükselir ancak dalgalanma da artar.",
    )

    # En yakın noktayı bul
    idx_closest = (frontier_df["Volatilite"] * 100 - user_target_vol).abs().idxmin()
    matched_row = frontier_df.loc[idx_closest]
    matched_weights = matched_row["Ağırlıklar"]

    u1, u2, u3 = st.columns(3)
    u1.metric("🎯 Seçtiğiniz Risk (Volatilite)", f"%{matched_row['Volatilite'] * 100:.2f}")
    u2.metric("📈 Elde Edeceğiniz Optimal Getiri", f"%{matched_row['Beklenen Getiri'] * 100:.2f}")
    u3.metric("⚡ Bu Noktadaki Sharpe Oranı", f"{matched_row['Sharpe']:.3f}")

    # Önerilen Sepet Ağırlıkları
    st.markdown("##### 📦 Bu Risk Seviyesi İçin Önerilen Hisse Sepeti Dağılımı:")
    
    df_user_weights = pd.DataFrame([
        {
            "Sembol": s,
            "Şirket": BIST_HISSELER.get(s, s),
            "Önerilen Pay (%)": round(w * 100, 2),
        }
        for s, w in matched_weights.items() if w > 0.005
    ]).sort_values(by="Önerilen Pay (%)", ascending=False)

    cu_bar, cu_table = st.columns([3, 2])
    
    with cu_bar:
        fig_user_bar = px.bar(
            df_user_weights,
            x="Sembol",
            y="Önerilen Pay (%)",
            text="Önerilen Pay (%)",
            color="Önerilen Pay (%)",
            color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig_user_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_user_bar.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig_user_bar, use_container_width=True)

    with cu_table:
        st.dataframe(df_user_weights, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# 5. ÇEŞİTLENDİRME (DIVERSIFICATION) MUCİZESİ
# =============================================================================
with st.expander("💡 Çeşitlendirme (Diversification) Mucizesi Nedir? Neden Sepet Yapmalıyız?"):
    st.markdown(f"""
    Grafikte dikkatinizi çekmiş olabilir: **Tekil hisselerin neredeyse tamamı mavi çizginin sağında ve altında kalır.**
    
    * **Örnek:** Portföydeki tek bir hisse örneğin **%{single_vols.max():.1f}** riskle işlem görürken; 
      hisseleri Markowitz formülüyle birleştirdiğinizde risk **%{min_vol['volatility'] * 100:.1f}** seviyesine kadar düşer!
    * **Nedeni:** Hisselerin fiyatları her gün aynı yönde ve aynı şiddette hareket etmez (kovaryans < 1). 
      Biri düşerken diğeri dengeler; böylece **getiriyi düşürmeden toplam riskinizi ciddi oranda azaltmış olursunuz**.
    """)

st.info("👉 Sol menüden **📈 4_Simulasyon_ve_CML** sayfasına giderek 2000+ portföylük Monte Carlo simülasyonunu ve Sermaye Pazarı Doğrusunu (CML) inceleyebilirsiniz.")
