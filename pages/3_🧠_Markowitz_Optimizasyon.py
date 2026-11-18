"""
3_🧠_Markowitz_Optimizasyon: Portföy Ağırlıkları, Etkin Sınır, Kullanıcı Özel Paneli ve İnteraktif Simülasyon
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
    portfolio_performance,
)

st.set_page_config(
    page_title="Markowitz Optimizasyonu & Özel Portföy Paneli — Markowitzz",
    page_icon="🧠",
    layout="wide",
)

# Özel CSS Tasarımı
st.markdown("""
<style>
    .user-panel {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 2px solid #F43F5E;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 25px;
    }
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
    .user-metric {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #F43F5E;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Markowitz Portföy Optimizasyonu & Özel Portföy Paneli")
st.markdown("""
Markowitz Modern Portföy Teorisi ile en ideal portföyleri hesaplayın, 
**kendi belirlediğiniz veya rastgele ürettiğiniz hisse ağırlıklarını test edin** ve sonucunuzun **Etkin Sınır (Efficient Frontier)** üzerindeki yerini anlık olarak görün!
""")

prices = st.session_state.get("prices")

if prices is None or prices.empty:
    st.warning("⚠️ Lütfen önce **📥 1_Veri_Cekme** sayfasına giderek hisse senedi verilerini çekin.")
    st.stop()

symbols = list(prices.columns)
n_assets = len(symbols)

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
    min_value=int(100 / n_assets),
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
# 1. 🛠️ KULLANICIYA ÖZEL PORTFÖY SİMÜLATÖRÜ & AĞIRLIK GİRİŞ PANELİ
# =============================================================================
st.markdown('<div class="user-panel">', unsafe_allow_html=True)
st.markdown("""
<h3 style="color: #F43F5E; margin-bottom: 5px;">🛠️ Kullanıcıya Özel Portföy Paneli</h3>
<p style="color: #CBD5E1; font-size: 0.95rem; margin-bottom: 15px;">
    Bu panelden çektiğiniz hisselere <b>kendi istediğiniz ağırlıkları girebilir</b> veya <b>rastgele ağırlıklar üreterek</b> 
    portföyünüzün getiri/risk noktasını aşağıdaki Etkin Sınır grafiğinde anlık olarak izleyebilirsiniz.
</p>
""", unsafe_allow_html=True)

# Session State Ağırlık Başlatma
if "custom_weights" not in st.session_state or len(st.session_state["custom_weights"]) != n_assets:
    eq_val = round(100.0 / n_assets, 1)
    st.session_state["custom_weights"] = {s: eq_val for s in symbols}

# Hızlı Butonlar
btn_c1, btn_c2, btn_c3, btn_space = st.columns([1.5, 1.5, 1.5, 3])

with btn_c1:
    if st.button("🎲 Rastgele Ağırlık Üret", use_container_width=True):
        # Rastgele Dirichlet dağılımı (toplam 100)
        rand_weights = np.random.dirichlet(np.ones(n_assets)) * 100.0
        st.session_state["custom_weights"] = {
            s: round(float(w), 1) for s, w in zip(symbols, rand_weights)
        }
        st.rerun()

with btn_c2:
    if st.button("⚖️ Eşit Paylaştır (1/N)", use_container_width=True):
        eq_val = round(100.0 / n_assets, 1)
        st.session_state["custom_weights"] = {s: eq_val for s in symbols}
        st.rerun()

with btn_c3:
    if st.button("🔄 %100'e Eşitle (Normalize)", use_container_width=True):
        cur_sum = sum(st.session_state["custom_weights"].values())
        if cur_sum > 0:
            st.session_state["custom_weights"] = {
                s: round((w / cur_sum) * 100.0, 1)
                for s, w in st.session_state["custom_weights"].items()
            }
            st.rerun()

st.markdown("##### ✏️ Hisse Ağırlıklarını Ayarlayın (%):")

# Hisse Ağırlık Slider'ları (Grid)
num_cols = min(4, max(2, n_assets))
cols = st.columns(num_cols)

for idx, sym in enumerate(symbols):
    with cols[idx % num_cols]:
        val = st.slider(
            f"{sym} ({BIST_HISSELER.get(sym, sym)[:12]}):",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state["custom_weights"].get(sym, round(100.0 / n_assets, 1))),
            step=1.0,
            key=f"slider_{sym}",
        )
        st.session_state["custom_weights"][sym] = val

# Toplam Ağırlık ve Normalizasyon
total_user_weight = sum(st.session_state["custom_weights"].values())
if total_user_weight > 0:
    normalized_user_w = np.array([
        st.session_state["custom_weights"][s] / total_user_weight for s in symbols
    ])
else:
    normalized_user_w = np.array([1.0 / n_assets] * n_assets)

# Kullanıcı Portföy Performansını Hesapla
user_ret, user_vol, user_sharpe = portfolio_performance(
    normalized_user_w,
    expected_returns.values,
    cov_matrix.values,
    scaled_rf,
)

# Kullanıcı Portföyü Gösterge Kartları
st.markdown("---")
u_col1, u_col2, u_col3, u_col4 = st.columns(4)

with u_col1:
    st.metric(
        "Girdiğiniz Toplam Ağırlık",
        f"%{total_user_weight:.1f}",
        delta="Dengeli (%100)" if abs(total_user_weight - 100) < 0.5 else f"%{total_user_weight - 100:+.1f} (Normalize edildi)",
        delta_color="normal" if abs(total_user_weight - 100) < 0.5 else "off",
    )

with u_col2:
    st.metric(
        f"Sizin Portföy Getiriniz ({scale_title})",
        f"%{user_ret * 100:.2f}",
        delta=f"%{(user_ret - eq_w['return']) * 100:+.2f} (Eşit Ağ. Kıyas)",
    )

with u_col3:
    st.metric(
        "Sizin Portföy Riskiniz (Volatilite)",
        f"%{user_vol * 100:.2f}",
        delta=f"%{(user_vol - min_vol['volatility']) * 100:+.2f} (Min Riski Fark)",
        delta_color="inverse",
    )

with u_col4:
    st.metric(
        "Sizin Sharpe Oranınız",
        f"{user_sharpe:.3f}",
        delta=f"{user_sharpe - max_sh['sharpe']:+.3f} (Max Sharpe Kıyas)",
    )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 2. TEMEL STRATEJİLER VE DAĞILIMLAR
# =============================================================================
st.subheader("🏆 Karşılaştırmalı Stratejiler")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="guide-card-green">
        <h4 style="color: #10B981; margin-bottom: 5px;">🥇 Maksimum Sharpe Portföyü</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">Risk başına en yüksek getiriyi sunan en akılcı ve dengeli portföy.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{max_sh['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{max_sh['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{max_sh['sharpe']:.3f}")

with c2:
    st.markdown("""
    <div class="guide-card">
        <h4 style="color: #38BDF8; margin-bottom: 5px;">🛡️ Minimum Volatilite Portföyü</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">"Geceleri rahat uyumak istiyorum" diyenler için dalgalanması en düşük sepet.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{min_vol['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{min_vol['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{min_vol['sharpe']:.3f}")

with c3:
    st.markdown("""
    <div class="guide-card-orange">
        <h4 style="color: #F59E0B; margin-bottom: 5px;">⚖️ Eşit Ağırlıklı Portföy (1/N)</h4>
        <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 5px;">"Hiç analiz yapmayıp her hisseye eşit pay verseydim ne olurdu?" kıyası.</p>
    </div>
    """, unsafe_allow_html=True)
    st.metric(f"Beklenen Getiri ({scale_title})", f"%{eq_w['return'] * 100:.2f}")
    st.metric("Volatilite (Risk)", f"%{eq_w['volatility'] * 100:.2f}")
    st.metric("Sharpe Oranı", f"{eq_w['sharpe']:.3f}")

# Dağılım Tablosu
weights_df = pd.DataFrame({
    "Sembol": symbols,
    "Şirket": [BIST_HISSELER.get(s, s) for s in symbols],
    "Sizin Portföyünüz (%)": (normalized_user_w * 100).round(2),
    "Max Sharpe Ağırlık (%)": (max_sh["weights"] * 100).round(2).values,
    "Min Volatilite Ağırlık (%)": (min_vol["weights"] * 100).round(2).values,
    "Eşit Ağırlık (%)": (eq_w["weights"] * 100).round(2).values,
})

with st.expander("📋 Tüm Portföy Dağılımlarını Tablo Olarak Gör", expanded=False):
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# 3. ETKİN SINIR (EFFICIENT FRONTIER) & KULLANICI PORTFÖYÜ GRAFİĞİ
# =============================================================================
st.subheader("📈 Etkin Sınır Üzerinde Portföyünüzün Yeri")
st.caption(f"Aşağıdaki grafikte pembe renkle gösterilen nokta (📍), yukarıda oluşturduğunuz **Sizin Portföyünüzdür**.")

# 3 Adımda Anlaşılır Kılavuz
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    <div class="guide-card">
        <h5 style="color: #38BDF8;">🧭 1. Hedef: Kuzeybatı</h5>
        <p style="color: #CBD5E1; font-size: 0.85rem;">
            Daha çok getiri (yukarı) ve daha az risk (sola). Portföyünüz ne kadar sol-üstteyse o kadar başarılıdır.
        </p>
    </div>
    """, unsafe_allow_html=True)

with g2:
    st.markdown("""
    <div class="guide-card-green">
        <h5 style="color: #10B981;">🚀 2. Mavi Eğri: Matematiksel Zirve</h5>
        <p style="color: #CBD5E1; font-size: 0.85rem;">
            Mavi çizgi, mevcut hisselerle ulaşılabilecek <b>en mükemmel sınırdır</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with g3:
    st.markdown("""
    <div class="guide-card-orange">
        <h5 style="color: #F59E0B;">📍 3. Sizin Noktanız Nerede?</h5>
        <p style="color: #CBD5E1; font-size: 0.85rem;">
            Eğer noktanız mavi çizgiye çok yakınsa sepetiniz verimlidir; altındaysa aynı riske daha çok getiri alabilirsiniz!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Etkin Sınır Verisi Hesapla
with st.spinner("Etkin Sınır eğrisi hesaplanıyor..."):
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
        name="Etkin Sınır (Mümkün Olan En İyi Portföyler)",
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
    marker=dict(size=10, color="#F59E0B", symbol="circle", line=dict(color="#FFFFFF", width=1)),
    customdata=single_sharpes.values,
    hovertemplate="<b>Hisse: %{text}</b><br>Risk (Volatilite): %{x:.2f}%<br>Beklenen Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
))

# 3. Maksimum Sharpe Noktası (Yeşil Yıldız)
fig_ef.add_trace(go.Scatter(
    x=[max_sh["volatility"] * 100],
    y=[max_sh["return"] * 100],
    mode="markers+text",
    text=["⭐ Max Sharpe"],
    textposition="bottom right",
    name="Max Sharpe (En Dengeli)",
    marker=dict(size=18, color="#10B981", symbol="star", line=dict(color="#FFFFFF", width=1.5)),
    hovertemplate="<b>Maksimum Sharpe Portföyü</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: " + f"{max_sh['sharpe']:.2f}<extra></extra>",
))

# 4. Minimum Volatilite Noktası (Kırmızı Elmas)
fig_ef.add_trace(go.Scatter(
    x=[min_vol["volatility"] * 100],
    y=[min_vol["return"] * 100],
    mode="markers+text",
    text=["🛡️ Min Volatilite"],
    textposition="top left",
    name="Min Volatilite (En Güvenli)",
    marker=dict(size=15, color="#EF4444", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
    hovertemplate="<b>Minimum Volatilite Portföyü</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: " + f"{min_vol['sharpe']:.2f}<extra></extra>",
))

# 5. Eşit Ağırlıklı Nokta (Mor Kare)
fig_ef.add_trace(go.Scatter(
    x=[eq_w["volatility"] * 100],
    y=[eq_w["return"] * 100],
    mode="markers+text",
    text=["⚖️ Eşit Dağılım"],
    textposition="bottom left",
    name="Eşit Dağılım (1/N)",
    marker=dict(size=13, color="#A855F7", symbol="square", line=dict(color="#FFFFFF", width=1)),
    hovertemplate="<b>Eşit Ağırlıklı Portföy</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<extra></extra>",
))

# 6. 📍 KULLANICI PORTFÖYÜ (ÖZEL VURGULU PEMBE NOKTA)
fig_ef.add_trace(go.Scatter(
    x=[user_vol * 100],
    y=[user_ret * 100],
    mode="markers+text",
    text=["📍 SİZİN PORTFÖYÜNÜZ"],
    textposition="top center",
    name="Sizin Portföyünüz",
    marker=dict(
        size=22,
        color="#F43F5E",
        symbol="circle-dot",
        line=dict(color="#FFFFFF", width=3),
    ),
    hovertemplate=(
        f"<b>📍 SİZİN PORTFÖYÜNÜZ</b><br>"
        f"Risk (Volatilite): %{user_vol * 100:.2f}%<br>"
        f"Beklenen Getiri: %{user_ret * 100:.2f}%<br>"
        f"Sharpe Oranı: {user_sharpe:.3f}<extra></extra>"
    ),
))

# Kullanıcının Noktasını İşaretleyen Ok
fig_ef.add_annotation(
    x=user_vol * 100,
    y=user_ret * 100,
    text="Sizin Portföyünüz Burada!",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.2,
    arrowcolor="#F43F5E",
    ax=-70,
    ay=-50,
    bgcolor="#1E293B",
    bordercolor="#F43F5E",
    font=dict(color="#F8FAFC", size=11, family="sans-serif"),
)

fig_ef.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (Standart Sapma) (%) — [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) — [{scale_title}]",
    hovermode="closest",
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_ef, use_container_width=True)

# Portföy Verimlilik Teşhisi
if not frontier_df.empty:
    # Kullanıcının riskiyle aynı veya en yakın etkin sınır getirisini bul
    idx_close_vol = (frontier_df["Volatilite"] - user_vol).abs().idxmin()
    best_ret_at_user_vol = frontier_df.loc[idx_close_vol, "Beklenen Getiri"]
    ret_gap = (best_ret_at_user_vol - user_ret) * 100

    if ret_gap <= 0.3:
        st.success(f"🎉 **Tebrikler!** Girdiğiniz portföy dağılımı Etkin Sınır'a neredeyse teğet (Getiri kaybı: sadece %{max(0.0, ret_gap):.2f}). Çok başarılı ve verimli bir sepet kurdunuz!")
    else:
        st.warning(f"💡 **Verimlilik Fırsatı:** Portföyünüz şu an Etkin Sınır'ın altında kalıyor. **Aynı risk seviyesinde (%{user_vol * 100:.2f})** kalarak Markowitz dağılımıyla **%{ret_gap:+.2f} daha fazla getiri** elde edebilirsiniz!")

st.markdown("---")

# =============================================================================
# 4. 🎯 RİSK KAYDIRMA ÇUBUĞU (ALTERNATİF ETKİN NOKTALAR)
# =============================================================================
with st.expander("🔍 Risk Seviyesine Göre Alternatif Optimal Noktaları Keşfet"):
    min_available_vol = float(frontier_df["Volatilite"].min() * 100)
    max_available_vol = float(frontier_df["Volatilite"].max() * 100)
    
    user_target_vol = st.slider(
        f"İncelemek İstediğiniz Risk Seviyesi (%) [{scale_title}]:",
        min_value=round(min_available_vol, 2),
        max_value=round(max_available_vol, 2),
        value=round(max_sh["volatility"] * 100, 2),
        step=0.25,
    )

    idx_closest = (frontier_df["Volatilite"] * 100 - user_target_vol).abs().idxmin()
    matched_row = frontier_df.loc[idx_closest]
    matched_weights = matched_row["Ağırlıklar"]

    st.markdown(f"**Hedeflenen Risk:** %{matched_row['Volatilite'] * 100:.2f} | **Optimal Getiri:** %{matched_row['Beklenen Getiri'] * 100:.2f} | **Sharpe:** {matched_row['Sharpe']:.3f}")
    
    df_opt_bar = pd.DataFrame([
        {"Sembol": s, "Önerilen Pay (%)": round(w * 100, 2)}
        for s, w in matched_weights.items() if w > 0.005
    ]).sort_values(by="Önerilen Pay (%)", ascending=False)
    
    st.dataframe(df_opt_bar, use_container_width=True, hide_index=True)

st.info("👉 Sol menüden **📈 4_Simulasyon_ve_CML** sayfasına giderek 2000+ portföylük Monte Carlo simülasyonunu ve Sermaye Pazarı Doğrusunu (CML) inceleyebilirsiniz.")
