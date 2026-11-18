"""
3_🧠_Markowitz_Optimizasyon: Etkin Sınır Üzerinde 5 Stratejik Portföy ve Özel Portföy Paneli
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
    get_five_frontier_portfolios,
    get_market_parameters,
    optimize_markowitz,
    portfolio_performance,
)

st.set_page_config(
    page_title="Markowitz Optimizasyonu & 5 Etkin Portföy — Markowitzz",
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
    .strategy-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
        height: 100%;
        text-align: center;
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
</style>
""", unsafe_allow_html=True)

st.title("🧠 Markowitz Optimizasyonu: Etkin Sınır Üzerinde 5 Stratejik Portföy")
st.markdown("""
Markowitz Etkin Sınırı (mavi eğri) üzerindeki **en düşük riskten en yüksek getiriye kadar uzanan 5 ideal portföy dağılımını** inceleyin. 
Ayrıca kendi belirlediğiniz özel portföyün bu 5 stratejiye göre nerede durduğunu canlı olarak test edin!
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

# 5 Stratejik Portföyü Hesapla (Doğrudan Etkin Sınır Üzerinde)
five_portfolios = get_five_frontier_portfolios(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=scaled_rf,
    max_weight=max_weight,
)

# Benchmark: Eşit Ağırlıklı Portföy
eq_weights = np.array([1.0 / n_assets] * n_assets)
eq_ret, eq_vol, eq_sharpe = portfolio_performance(eq_weights, expected_returns.values, cov_matrix.values, scaled_rf)

# =============================================================================
# 1. 🎯 ETKİN SINIR ÜZERİNDEKİ 5 STRATEJİK PORTFÖY ÖZETİ
# =============================================================================
st.subheader("🎯 Etkin Sınır Üzerinde 5 Stratejik Portföy")
st.markdown("""
Aşağıdaki 5 portföy, risk toleransınıza göre doğrudan **Etkin Sınır (mavi eğri)** üzerinden seçilmiş en ideal matematiksel noktalardır:
""")

cols_5 = st.columns(5)
for idx, p in enumerate(five_portfolios):
    with cols_5[idx]:
        st.markdown(f"""
        <div class="strategy-card" style="border-top: 4px solid {p['color']};">
            <h4 style="color: {p['color']}; margin-bottom: 2px;">{p['id']}</h4>
            <div style="font-size: 0.8rem; font-weight: 600; color: #CBD5E1; margin-bottom: 8px;">{p['tag']}</div>
            <div style="font-size: 0.78rem; color: #94A3B8; min-height: 40px; margin-bottom: 12px;">{p['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric(f"Getiri ({scale_title})", f"%{p['return'] * 100:.2f}")
        st.metric("Risk (Volatilite)", f"%{p['volatility'] * 100:.2f}")
        st.metric("Sharpe Oranı", f"{p['sharpe']:.3f}")

st.markdown("---")

# =============================================================================
# 2. 🛠️ KULLANICIYA ÖZEL PORTFÖY PANELİ
# =============================================================================
st.markdown('<div class="user-panel">', unsafe_allow_html=True)
st.markdown("""
<h3 style="color: #F43F5E; margin-bottom: 5px;">🛠️ Kullanıcıya Özel Portföy Paneli</h3>
<p style="color: #CBD5E1; font-size: 0.95rem; margin-bottom: 15px;">
    Hisselere <b>kendi istediğiniz ağırlıkları verin</b> veya <b>rastgele dağıtın</b>. 
    Portföyünüzün bu 5 stratejiye ve Etkin Sınır'a göre yerini aşağıdaki grafikte <b>pembe nokta (📍)</b> olarak izleyin!
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
u_col1.metric(
    "Girdiğiniz Toplam Ağırlık",
    f"%{total_user_weight:.1f}",
    delta="Dengeli (%100)" if abs(total_user_weight - 100) < 0.5 else f"%{total_user_weight - 100:+.1f} (Normalize edildi)",
    delta_color="normal" if abs(total_user_weight - 100) < 0.5 else "off",
)
u_col2.metric(f"Sizin Portföy Getiriniz ({scale_title})", f"%{user_ret * 100:.2f}")
u_col3.metric("Sizin Portföy Riskiniz (Volatilite)", f"%{user_vol * 100:.2f}")
u_col4.metric("Sizin Sharpe Oranınız", f"{user_sharpe:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 3. ETKİN SINIR GRAFİĞİ (5 PORTFÖY + KULLANICI PORTFÖYÜ)
# =============================================================================
st.subheader("📈 Etkin Sınır Grafiği & 5 Portföyün Konumu")
st.caption("Aşağıdaki grafikte mavi çizgi üzerindeki 5 renkli nokta (P1 - P5) stratejik portföyleri, pembe nokta (📍) ise sizin özel portföyünüzü gösterir.")

with st.spinner("Etkin Sınır eğrisi ve noktalar hesaplanıyor..."):
    frontier_df = generate_efficient_frontier(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=scaled_rf,
        max_weight=max_weight,
        points=60,
    )

fig_ef = go.Figure()

# 1. Etkin Sınır Eğrisi
if not frontier_df.empty:
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["Volatilite"] * 100,
        y=frontier_df["Beklenen Getiri"] * 100,
        mode="lines",
        name="Etkin Sınır (Mavi Eğri)",
        line=dict(color="#38BDF8", width=3.5),
        customdata=frontier_df["Sharpe"],
        hovertemplate="<b>Etkin Sınır Noktası</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
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
    marker=dict(size=9, color="#94A3B8", symbol="circle", opacity=0.7),
    customdata=single_sharpes.values,
    hovertemplate="<b>Hisse: %{text}</b><br>Risk: %{x:.2f}%<br>Getiri: %{y:.2f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
))

# 3. ETKİN SINIR ÜZERİNDEKİ 5 STRATEJİK PORTFÖYÜ ÇİZ
for p in five_portfolios:
    fig_ef.add_trace(go.Scatter(
        x=[p["volatility"] * 100],
        y=[p["return"] * 100],
        mode="markers+text",
        text=[f"<b>{p['id']}</b>"],
        textposition="top right",
        name=p["name"],
        marker=dict(
            size=17,
            color=p["color"],
            symbol=p["symbol"],
            line=dict(color="#FFFFFF", width=2),
        ),
        hovertemplate=(
            f"<b>{p['name']}</b> ({p['tag']})<br>"
            f"Risk (Volatilite): %{p['volatility'] * 100:.2f}%<br>"
            f"Beklenen Getiri: %{p['return'] * 100:.2f}%<br>"
            f"Sharpe Oranı: {p['sharpe']:.3f}<extra></extra>"
        ),
    ))

# 4. KULLANICI PORTFÖYÜ (📍 PEMBE NOKTA)
fig_ef.add_trace(go.Scatter(
    x=[user_vol * 100],
    y=[user_ret * 100],
    mode="markers+text",
    text=["📍 SİZİN PORTFÖYÜNÜZ"],
    textposition="bottom center",
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

fig_ef.update_layout(
    template="plotly_dark",
    xaxis_title=f"Volatilite / Risk (%) — [{scale_title}]",
    yaxis_title=f"Beklenen Getiri (%) — [{scale_title}]",
    hovermode="closest",
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_ef, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4. 5 PORTFÖYÜN HİSSE DAĞILIMLARI KARŞILAŞTIRMA TABLOSU & GRAFİĞİ
# =============================================================================
st.subheader("📊 5 Portföyün Hisse Dağılım Tablosu & Karşılaştırması")

# Karşılaştırma DataFrame'i
comp_dict = {
    "Sembol": symbols,
    "Şirket": [BIST_HISSELER.get(s, s) for s in symbols],
}
for p in five_portfolios:
    comp_dict[f"{p['id']} ({p['short_name'].split(':')[1].strip()}) (%)"] = (p["weights"] * 100).round(2).values

comp_dict["Sizin Portföyünüz (%)"] = (normalized_user_w * 100).round(2)

df_all_weights = pd.DataFrame(comp_dict)
st.dataframe(df_all_weights, use_container_width=True, hide_index=True)

# Gruplanmış Hisse Ağırlık Çubuk Grafiği
st.markdown("##### 📈 Risk Seviyesi Arttıkça Hisse Ağırlıklarının Değişimi:")
st.caption("P1'den P5'e doğru gidildikçe hisse dağılımlarının defansiften agresife nasıl evrildiğini inceleyin:")

melted_weights = []
for p in five_portfolios:
    for sym, w in p["weights"].items():
        if w > 0.005:
            melted_weights.append({
                "Portföy": p["short_name"],
                "Hisse": sym,
                "Ağırlık (%)": round(w * 100, 2),
            })

df_melted = pd.DataFrame(melted_weights)
fig_grouped_bar = px.bar(
    df_melted,
    x="Portföy",
    y="Ağırlık (%)",
    color="Hisse",
    title="Etkin Sınır Üzerindeki 5 Portföyün Hisse Kırılımı",
    template="plotly_dark",
    text="Ağırlık (%)",
)
fig_grouped_bar.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
fig_grouped_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), barmode="stack")
st.plotly_chart(fig_grouped_bar, use_container_width=True)

st.info("👉 Sol menüden **📈 4_Simulasyon_ve_CML** sayfasına giderek 2000+ portföylük Monte Carlo simülasyonunu ve Sermaye Pazarı Doğrusunu (CML) inceleyebilirsiniz.")
