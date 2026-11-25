"""
3_🧠_Markowitz_Optimizasyon: Tam Parabol, Min Vol Altındaki Portföyler (A1-A4), 
10 Stratejik Portföy (P1-P10) ve Özel Portföy Paneli
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
    get_ten_frontier_portfolios,
    portfolio_performance,
)

st.set_page_config(
    page_title="Markowitz Optimizasyonu & Tam Parabol Portföyleri — Markowitzz",
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
    .group-card-lower {
        background: #1E293B;
        border-radius: 12px;
        padding: 14px 18px;
        border-left: 4px solid #94A3B8;
        height: 100%;
    }
    .group-card-below {
        background: #1E293B;
        border-radius: 12px;
        padding: 14px 18px;
        border-left: 4px solid #06B6D4;
        height: 100%;
    }
    .group-card-center {
        background: #1E293B;
        border-radius: 12px;
        padding: 14px 18px;
        border-left: 4px solid #10B981;
        height: 100%;
    }
    .group-card-above {
        background: #1E293B;
        border-radius: 12px;
        padding: 14px 18px;
        border-left: 4px solid #F43F5E;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Markowitz Optimizasyonu: Tam Parabol & Alt Kanat Portföyleri")
st.markdown("""
Markowitz parabolü üzerindeki tüm bölgeleri inceleyin: **Minimum Volatilitenin altındaki alt kanat portföyleri (A1 - A4)**, 
**Sharpe altındaki temkinli portföyler (P1 - P5)** ve **Sharpe üstündeki getiri odaklı portföyler (P6 - P10)**!
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

# Parabol Üzerindeki Portföyleri Hesapla (A1..A4, P1..P5, MS, P6..P10)
ten_portfolios, max_sharpe_info, lower_portfolios = get_ten_frontier_portfolios(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=scaled_rf,
    max_weight=max_weight,
)

portfolios_below = [p for p in ten_portfolios if "Altı" in p["group"]]
portfolios_above = [p for p in ten_portfolios if "Üstü" in p["group"]]

# =============================================================================
# 1. PARABOLÜN 4 BÖLGESİ ÖZET KARTLARI
# =============================================================================
b0, b1, b2, b3 = st.columns(4)

with b0:
    st.markdown("""
    <div class="group-card-lower">
        <h5 style="color: #94A3B8; margin-bottom: 2px;">⚠️ Min Vol Altı (A1 - A4)</h5>
        <p style="color: #CBD5E1; font-size: 0.8rem; margin-bottom: 0;">
            <b>Alt Kanat (Verimsiz):</b> P1'in altındaki portföyler. Aynı riske daha az getiri verir.
        </p>
    </div>
    """, unsafe_allow_html=True)

with b1:
    st.markdown("""
    <div class="group-card-below">
        <h5 style="color: #06B6D4; margin-bottom: 2px;">🛡️ Sharpe Altı (P1 - P5)</h5>
        <p style="color: #CBD5E1; font-size: 0.8rem; margin-bottom: 0;">
            <b>Temkinli / Defansif:</b> P1 Min Volatilite ile Maksimum Sharpe arasındaki düşük riskli sepetler.
        </p>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="group-card-center">
        <h5 style="color: #10B981; margin-bottom: 2px;">⭐ Max Sharpe (Zirve)</h5>
        <p style="color: #CBD5E1; font-size: 0.8rem; margin-bottom: 0;">
            <b>Teğet Noktası:</b> Risk başına en yüksek birim getiri (Getiri: %{max_sharpe_info['return']*100:.1f}).
        </p>
    </div>
    """, unsafe_allow_html=True)

with b3:
    st.markdown("""
    <div class="group-card-above">
        <h5 style="color: #F43F5E; margin-bottom: 2px;">🚀 Sharpe Üstü (P6 - P10)</h5>
        <p style="color: #CBD5E1; font-size: 0.8rem; margin-bottom: 0;">
            <b>Dinamik / Agresif:</b> Yüksek getiri hedefleyen, riski artırılmış atak portföyler.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 2. 🛠️ KULLANICIYA ÖZEL PORTFÖY PANELİ
# =============================================================================
st.markdown('<div class="user-panel">', unsafe_allow_html=True)
st.markdown("""
<h3 style="color: #F43F5E; margin-bottom: 5px;">🛠️ Kullanıcıya Özel Portföy Paneli</h3>
<p style="color: #CBD5E1; font-size: 0.95rem; margin-bottom: 15px;">
    Hisselere <b>kendi istediğiniz ağırlıkları verin</b> veya <b>rastgele dağıtın</b>. 
    Portföyünüzün bu stratejilere göre yerini aşağıdaki grafikte <b>pembe nokta (📍)</b> olarak izleyin!
</p>
""", unsafe_allow_html=True)

if "custom_weights" not in st.session_state or len(st.session_state["custom_weights"]) != n_assets:
    eq_val = round(100.0 / n_assets, 1)
    st.session_state["custom_weights"] = {s: eq_val for s in symbols}

btn_c1, btn_c2, btn_c3, btn_space = st.columns([1.5, 1.5, 1.5, 3])

with btn_c1:
    if st.button("🎲 Rastgele Ağırlık Üret", use_container_width=True):
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

st.markdown("##### ✏️ Hisse Ağırlıklarını Belirleyin (%):")

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

total_user_weight = sum(st.session_state["custom_weights"].values())
if total_user_weight > 0:
    normalized_user_w = np.array([
        st.session_state["custom_weights"][s] / total_user_weight for s in symbols
    ])
else:
    normalized_user_w = np.array([1.0 / n_assets] * n_assets)

user_ret, user_vol, user_sharpe = portfolio_performance(
    normalized_user_w,
    expected_returns.values,
    cov_matrix.values,
    scaled_rf,
)

u_col1, u_col2, u_col3, u_col4 = st.columns(4)
u_col1.metric("Girdiğiniz Toplam Ağırlık", f"%{total_user_weight:.1f}")
u_col2.metric(f"Sizin Portföy Getiriniz ({scale_title})", f"%{user_ret * 100:.2f}")
u_col3.metric("Sizin Portföy Riskiniz (Volatilite)", f"%{user_vol * 100:.2f}")
u_col4.metric("Sizin Sharpe Oranınız", f"{user_sharpe:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 3. ETKİN SINIR GRAFİĞİ (TAM PARABOL: A1..A4 + P1..P10 + MS + KULLANICI)
# =============================================================================
st.subheader("📈 Tam Markowitz Parabolü Üzerinde Portföylerin Dağılımı")
st.caption("Aşağıdaki grafikte mavi çizgi tam parabolü oluşturur: P1'in altında kalan gri halkalar (A1-A4), üst kanattaki P1-P10 stratejileri ve sizin noktanız (📍).")

with st.spinner("Tam parabol eğrisi ve stratejik noktalar hesaplanıyor..."):
    all_reference_points = ten_portfolios + [max_sharpe_info] + lower_portfolios
    frontier_df = generate_efficient_frontier(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=scaled_rf,
        max_weight=max_weight,
        points=100,
        include_portfolios=all_reference_points,
    )

fig_ef = go.Figure()

# 1. Tam Parabol Eğrisi
if not frontier_df.empty:
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["Volatilite"] * 100,
        y=frontier_df["Beklenen Getiri"] * 100,
        mode="lines",
        name="Markowitz Parabolü",
        line=dict(color="#38BDF8", width=3.5),
        customdata=frontier_df["Sharpe"],
        hovertemplate="<b>Markowitz Parabolü</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
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
    marker=dict(size=9, color="#64748B", symbol="circle", opacity=0.55),
    customdata=single_sharpes.values,
    hovertemplate="<b>Hisse: %{text}</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
))

# 3. ⚠️ MİNİMUM VOLATİLİTE ALTINDAKİ PORTFÖYLER (A1 - A4: ALT KANAT)
for p in lower_portfolios:
    fig_ef.add_trace(go.Scatter(
        x=[p["volatility"] * 100],
        y=[p["return"] * 100],
        mode="markers+text",
        text=[f"<b>{p['id']}</b>"],
        textposition="bottom right",
        name=f"{p['id']} (Alt Kanat)",
        marker=dict(size=13, color=p["color"], symbol="circle-open", line=dict(color="#CBD5E1", width=2)),
        hovertemplate=(
            f"<b>{p['name']}</b><br>"
            f"Grup: {p['group']}<br>"
            f"Risk: %{p['volatility'] * 100:.2f}%<br>"
            f"Getiri: %{p['return'] * 100:.2f}%<br>"
            f"Sharpe: {p['sharpe']:.3f}<br>"
            "<i>⚠️ Min Volatilitenin altında kalan verimsiz sepet</i><extra></extra>"
        ),
    ))

# 4. 🛡️ SHARPE ALTINDAKİ 5 PORTFÖY (P1 - P5)
for p in portfolios_below:
    fig_ef.add_trace(go.Scatter(
        x=[p["volatility"] * 100],
        y=[p["return"] * 100],
        mode="markers+text",
        text=[f"<b>{p['id']}</b>"],
        textposition="top left",
        name=f"{p['id']} (Sharpe Altı)",
        marker=dict(size=14, color=p["color"], symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate=(
            f"<b>{p['name']}</b><br>"
            f"Grup: {p['group']}<br>"
            f"Risk: %{p['volatility'] * 100:.2f}%<br>"
            f"Getiri: %{p['return'] * 100:.2f}%<br>"
            f"Sharpe: {p['sharpe']:.3f}<extra></extra>"
        ),
    ))

# 5. ⭐ MERKEZ REFERANS: MAKSİMUM SHARPE NOKTASI
fig_ef.add_trace(go.Scatter(
    x=[max_sharpe_info["volatility"] * 100],
    y=[max_sharpe_info["return"] * 100],
    mode="markers+text",
    text=["<b>⭐ MAX SHARPE</b>"],
    textposition="bottom right",
    name="⭐ Max Sharpe (Zirve)",
    marker=dict(size=20, color="#10B981", symbol="star", line=dict(color="#FFFFFF", width=2)),
    hovertemplate=(
        f"<b>⭐ MAKSİMUM SHARPE (ZİRVE)</b><br>"
        f"Risk: %{max_sharpe_info['volatility'] * 100:.2f}%<br>"
        f"Getiri: %{max_sharpe_info['return'] * 100:.2f}%<br>"
        f"Sharpe: {max_sharpe_info['sharpe']:.3f}<extra></extra>"
    ),
))

# 6. 🚀 SHARPE ÜSTÜNDEKİ 5 PORTFÖY (P6 - P10)
for p in portfolios_above:
    fig_ef.add_trace(go.Scatter(
        x=[p["volatility"] * 100],
        y=[p["return"] * 100],
        mode="markers+text",
        text=[f"<b>{p['id']}</b>"],
        textposition="top right",
        name=f"{p['id']} (Sharpe Üstü)",
        marker=dict(size=14, color=p["color"], symbol="triangle-up", line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate=(
            f"<b>{p['name']}</b><br>"
            f"Grup: {p['group']}<br>"
            f"Risk: %{p['volatility'] * 100:.2f}%<br>"
            f"Getiri: %{p['return'] * 100:.2f}%<br>"
            f"Sharpe: {p['sharpe']:.3f}<extra></extra>"
        ),
    ))

# 7. 📍 KULLANICI PORTFÖYÜ
fig_ef.add_trace(go.Scatter(
    x=[user_vol * 100],
    y=[user_ret * 100],
    mode="markers+text",
    text=["📍 SİZİN PORTFÖYÜNÜZ"],
    textposition="bottom center",
    name="Sizin Portföyünüz",
    marker=dict(size=22, color="#F43F5E", symbol="circle-dot", line=dict(color="#FFFFFF", width=3)),
    hovertemplate=(
        f"<b>📍 SİZİN PORTFÖYÜNÜZ</b><br>"
        f"Risk: %{user_vol * 100:.2f}%<br>"
        f"Getiri: %{user_ret * 100:.2f}%<br>"
        f"Sharpe: {user_sharpe:.3f}<extra></extra>"
    ),
))

fig_ef.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (%) — [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) — [{scale_title}]",
    hovermode="closest",
    height=640,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_ef, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. PORTFÖYLERİN KARŞILAŞTIRMA TABLOSU VE İNCELEMESİ
# =============================================================================
st.subheader("📋 Parabol Üzerindeki Tüm Portföylerin Tablosu")

tab_choice = st.radio(
    "Filtrele:",
    [
        "Tüm Portföyler (A1-A4 + P1-P10 + Max Sharpe)",
        "⚠️ Sadece Min Vol Altı (A1 - A4: Alt Kanat)",
        "🛡️ Sadece Sharpe Altı (P1 - P5)",
        "🚀 Sadece Sharpe Üstü (P6 - P10)",
    ],
    horizontal=True,
)

if "Min Vol Altı" in tab_choice:
    displayed_list = lower_portfolios
elif "Sharpe Altı" in tab_choice:
    displayed_list = portfolios_below
elif "Sharpe Üstü" in tab_choice:
    displayed_list = portfolios_above
else:
    displayed_list = lower_portfolios + portfolios_below + [max_sharpe_info] + portfolios_above

summary_rows = []
for p in displayed_list:
    top_stocks = p["weights"][p["weights"] > 0.05].sort_values(ascending=False)
    top_str = ", ".join([f"{s} (%{w*100:.0f})" for s, w in top_stocks.items()])
    summary_rows.append({
        "Kod": p["id"],
        "Portföy Stratejisi": p["name"],
        "Grup / Tanım": p["tag"],
        f"Beklenen Getiri (%)": round(p["return"] * 100, 2),
        f"Risk (Volatilite) (%)": round(p["volatility"] * 100, 2),
        "Sharpe Oranı": round(p["sharpe"], 3),
        "Ağırlıklı Hisseler": top_str,
    })

df_summary = pd.DataFrame(summary_rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

with st.expander("💡 Finansal Not: Min Volatilitenin Altındaki (A1 - A4) Portföyler Neden 'Verimsiz' Sayılır?"):
    st.markdown(f"""
    Markowitz Portföy Teorisinde **P1 (Minimum Volatilite)** portföyü parabolün dönüm noktasıdır (burun noktası).
    
    * **Örnek Kıyas:** Tablodan **A2** portföyü ile üst kanattaki **P3** portföyünü karşılaştırın:
      * İki portföy de neredeyse **aynı riske** (yaklaşık %{portfolios_below[2]['volatility']*100:.1f} volatilite) sahiptir.
      * Ancak **P3 portföyü %{portfolios_below[2]['return']*100:.1f}** getiri sağlarken, alt kanattaki **A2 portföyü sadece %{lower_portfolios[1]['return']*100:.1f}** getiri sağlar!
    * **Sonuç:** Rasyonel bir yatırımcı aynı riski alıp daha az kazanmak istemeyeceği için, P1'in altında kalan bu portföylere **"Baskılanmış / Verimsiz Portföyler"** adı verilir.
    """)

# Ağırlık Karşılaştırma Matrisi
with st.expander("📊 Tüm Portföylerin Hisse Yüzdelerini Yan Yana Gör"):
    comp_dict = {
        "Sembol": symbols,
        "Şirket": [BIST_HISSELER.get(s, s) for s in symbols],
    }
    for p in lower_portfolios:
        comp_dict[f"{p['id']} (%)"] = (p["weights"] * 100).round(1).values
    for p in ten_portfolios:
        comp_dict[f"{p['id']} (%)"] = (p["weights"] * 100).round(1).values
    comp_dict["⭐ Max Sharpe (%)"] = (max_sharpe_info["weights"] * 100).round(1).values
    comp_dict["📍 Sizin (%)"] = (normalized_user_w * 100).round(1)

    df_full_weights = pd.DataFrame(comp_dict)
    st.dataframe(df_full_weights, use_container_width=True, hide_index=True)

st.info("👉 Sol menüden **📈 4_Simulasyon_ve_CML** sayfasına giderek 2000+ portföylük Monte Carlo simülasyonunu ve Sermaye Pazarı Doğrusunu (CML) inceleyebilirsiniz.")
