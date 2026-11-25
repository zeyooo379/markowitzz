"""
Markowitz Portföy Optimizasyon Motoru
Dönemsel (Kullanıcının seçtiği T gün) ve Yıllıklaştırılmış (252 gün) hesaplamaları destekler.
Tam Parabol (Alt Kanat + Etkin Üst Kanat) ve 10 Stratejik Portföy (5 Sharpe Altı, 5 Sharpe Üstü).
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_returns(prices: pd.DataFrame, return_type: str = "Basit Getiri") -> pd.DataFrame:
    """Fiyatlardan getiri serilerini hesaplar."""
    if return_type == "Basit Getiri":
        returns = prices.pct_change(fill_method=None).dropna()
    else:
        returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def get_market_parameters(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    mode: str = "period",
    annual_trading_days: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Kovaryans, getiri ve risk parametrelerini seçilen ölçeğe göre hesaplar.
    """
    actual_days = len(returns)
    if actual_days == 0:
        raise ValueError("Getiri serisi boş olamaz.")

    daily_mean = returns.mean()
    daily_cov = returns.cov()
    daily_std = returns.std()
    correlation = returns.corr()

    if not prices.empty:
        cumulative_period_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    else:
        cumulative_period_return = daily_mean * actual_days

    if mode == "period":
        time_factor = float(actual_days)
        scale_label = f"Seçilen Dönem ({actual_days} İşlem Günü)"
        period_rf = (1 + annual_risk_free_rate) ** (actual_days / annual_trading_days) - 1
    else:
        time_factor = float(annual_trading_days)
        scale_label = f"Yıllıklaştırılmış ({annual_trading_days} İşlem Günü Projeksiyonu)"
        period_rf = annual_risk_free_rate

    expected_returns = daily_mean * time_factor
    covariance_matrix = daily_cov * time_factor
    volatilities = daily_std * np.sqrt(time_factor)

    return {
        "mode": mode,
        "actual_days": actual_days,
        "time_factor": time_factor,
        "scale_label": scale_label,
        "expected_returns": expected_returns,
        "cumulative_returns": cumulative_period_return,
        "covariance_matrix": covariance_matrix,
        "volatilities": volatilities,
        "correlation_matrix": correlation,
        "daily_covariance": daily_cov,
        "risk_free_rate": period_rf,
        "annual_risk_free_rate": annual_risk_free_rate,
        "annual_trading_days": annual_trading_days,
    }


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """Portföy beklenen getirisi, volatilitesi ve Sharpe oranını döndürür."""
    weights = np.array(weights)
    p_return = float(np.dot(weights, expected_returns))
    p_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_volatility = float(np.sqrt(max(p_var, 1e-12)))
    
    sharpe = (p_return - risk_free_rate) / p_volatility if p_volatility > 0 else 0.0
    return p_return, p_volatility, sharpe


def optimize_markowitz(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> Dict[str, Any]:
    """
    Markowitz Optimizasyonu:
    - Minimum Volatilite Portföyü
    - Maksimum Sharpe Portföyü (Teğet Portföy)
    - Eşit Ağırlıklı Portföy (Benchmark)
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_weights = np.array([1.0 / n_assets] * n_assets)

    # 1. Minimum Volatilite Portföyü
    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    res_min_vol = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    w_min_vol = np.clip(res_min_vol.x if res_min_vol.success else init_weights, 0, max_weight)
    w_min_vol /= np.sum(w_min_vol)
    ret_min_vol, vol_min_vol, sh_min_vol = portfolio_performance(w_min_vol, mu, sigma, risk_free_rate)

    # 2. Maksimum Sharpe Portföyü
    def neg_sharpe_obj(w):
        r, v, s = portfolio_performance(w, mu, sigma, risk_free_rate)
        return -s

    res_max_sh = minimize(neg_sharpe_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    w_max_sh = np.clip(res_max_sh.x if res_max_sh.success else init_weights, 0, max_weight)
    w_max_sh /= np.sum(w_max_sh)
    ret_max_sh, vol_max_sh, sh_max_sh = portfolio_performance(w_max_sh, mu, sigma, risk_free_rate)

    # 3. Eşit Ağırlıklı Portföy (1/N)
    w_eq = init_weights
    ret_eq, vol_eq, sh_eq = portfolio_performance(w_eq, mu, sigma, risk_free_rate)

    return {
        "min_volatility": {
            "weights": pd.Series(w_min_vol, index=symbols),
            "return": ret_min_vol,
            "volatility": vol_min_vol,
            "sharpe": sh_min_vol,
        },
        "max_sharpe": {
            "weights": pd.Series(w_max_sh, index=symbols),
            "return": ret_max_sh,
            "volatility": vol_max_sh,
            "sharpe": sh_max_sh,
        },
        "equal_weight": {
            "weights": pd.Series(w_eq, index=symbols),
            "return": ret_eq,
            "volatility": vol_eq,
            "sharpe": sh_eq,
        },
    }


def _solve_frontier_point(
    target_r: float,
    start_w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    bounds: tuple,
    max_weight: float,
    risk_free_rate: float,
) -> Tuple[np.ndarray, float, float, float]:
    """Belirli bir hedef getiri için minimum volatilite portföyünü çözer."""
    sum_c = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    c = (sum_c, {"type": "eq", "fun": lambda w: np.dot(w, mu) - target_r})

    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    best_res = None
    best_vol = float("inf")

    init_equal = np.array([1.0 / len(mu)] * len(mu))
    for x0 in [start_w, init_equal]:
        res = minimize(
            min_vol_obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=c,
            options={"ftol": 1e-9, "maxiter": 400},
        )
        if res.success:
            w = np.clip(res.x, 0, max_weight)
            w /= np.sum(w)
            ret, vol, sh = portfolio_performance(w, mu, sigma, risk_free_rate)
            if vol < best_vol:
                best_vol = vol
                best_res = (w, ret, vol, sh)

    if best_res is not None:
        return best_res

    # Ceza fonksiyonlu fallback
    def penalized_obj(w):
        return min_vol_obj(w) + 1500.0 * (np.dot(w, mu) - target_r) ** 2

    res_pen = minimize(
        penalized_obj,
        start_w,
        method="SLSQP",
        bounds=bounds,
        constraints=sum_c,
        options={"ftol": 1e-9, "maxiter": 400},
    )
    w_pen = np.clip(res_pen.x, 0, max_weight)
    w_pen /= np.sum(w_pen)
    ret, vol, sh = portfolio_performance(w_pen, mu, sigma, risk_free_rate)
    return w_pen, ret, vol, sh


def get_ten_frontier_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Etkin Sınır (Efficient Frontier) parabolü üzerinde tam 10 stratejik portföy hesaplar:
    - 5 Tanesi Maksimum Sharpe'ın Altında (P1 - P5)
    - 5 Tanesi Maksimum Sharpe'ın Üstünde (P6 - P10)
    Tüm portföyler (P6, P7, P8 dahil) parabol çizgisinin birebir üstünde oturur.
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    init_weights = np.array([1.0 / n_assets] * n_assets)
    sum_c = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    # 1. Global Minimum Volatilite (P1 / burun noktası)
    res_min = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_min = np.clip(res_min.x if res_min.success else init_weights, 0, max_weight)
    w_min /= np.sum(w_min)
    r_min, v_min, s_min = portfolio_performance(w_min, mu, sigma, risk_free_rate)

    # 2. Kısıtlar altında Maksimum Mümkün Getiri (P10 / tepe noktası)
    def neg_ret_obj(w):
        return -np.dot(w, mu)

    res_max = minimize(neg_ret_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_max = np.clip(res_max.x if res_max.success else init_weights, 0, max_weight)
    w_max /= np.sum(w_max)
    r_max, v_max, s_max = portfolio_performance(w_max, mu, sigma, risk_free_rate)

    # 3. Maksimum Sharpe (Teğet Noktası)
    def neg_sharpe_obj(w):
        r, v, s = portfolio_performance(w, mu, sigma, risk_free_rate)
        return -s

    res_sh = minimize(neg_sharpe_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_sh = np.clip(res_sh.x if res_sh.success else init_weights, 0, max_weight)
    w_sh /= np.sum(w_sh)
    r_sh, v_sh, s_sh = portfolio_performance(w_sh, mu, sigma, risk_free_rate)

    # Tutarlılık kontrolü
    if r_sh <= r_min:
        r_sh = r_min + (r_max - r_min) * 0.4
    if r_max <= r_sh + 1e-4:
        r_max = r_sh + 0.01

    # 5 Nokta Sharpe Altında (r_min'den r_sh'ye 5 nokta)
    targets_below = list(np.linspace(r_min, r_sh, 6)[:-1])
    # 5 Nokta Sharpe Üstünde (r_sh'den r_max'a 5 nokta)
    targets_above = list(np.linspace(r_sh, r_max, 6)[1:])

    below_names = [
        ("P1", "P1: Min Volatilite (Çok Düşük Risk)", "🛡️ En Düşük Risk", "#06B6D4"),
        ("P2", "P2: Defansif Artı", "🌱 Çok Temkinli", "#0EA5E9"),
        ("P3", "P3: Muhafazakar Dengeli", "⚖️ Dengeli Muhafazakar", "#3B82F6"),
        ("P4", "P4: Temkinli Büyüme", "📈 Düzenli Büyüme", "#6366F1"),
        ("P5", "P5: Dengeli Adayı", "🎯 Sharpe'a Yakın Denge", "#8B5CF6"),
    ]

    above_names = [
        ("P6", "P6: Dinamik Büyüme", "🚀 İleri Büyüme", "#EC4899"),
        ("P7", "P7: Yüksek Getiri Odaklı", "⚡ Yüksek Potansiyel", "#F43F5E"),
        ("P8", "P8: Atak Portföy", "🔥 Atak Büyüme", "#F97316"),
        ("P9", "P9: Yüksek Risk / Getiri", "💥 Yüksek Riskli", "#EA580C"),
        ("P10", "P10: Maksimum Getiri (Agresif)", "🏆 Zirve Kazanç", "#DC2626"),
    ]

    ten_portfolios = []

    # Sharpe Altı Portföyler (Warm-start ile çöz)
    curr_w = w_min
    for (pid, pname, ptag, pcolor), t_r in zip(below_names, targets_below):
        w_p, ret_p, vol_p, sh_p = _solve_frontier_point(t_r, curr_w, mu, sigma, bounds, max_weight, risk_free_rate)
        curr_w = w_p
        ten_portfolios.append({
            "id": pid,
            "name": pname,
            "tag": ptag,
            "group": "Düşük Risk Bölgesi (Sharpe Altı)",
            "color": pcolor,
            "symbol": "diamond",
            "return": ret_p,
            "volatility": vol_p,
            "sharpe": sh_p,
            "weights": pd.Series(w_p, index=symbols),
        })

    # Sharpe Üstü Portföyler (Warm-start ile çöz)
    curr_w = w_sh
    for (pid, pname, ptag, pcolor), t_r in zip(above_names, targets_above):
        w_p, ret_p, vol_p, sh_p = _solve_frontier_point(t_r, curr_w, mu, sigma, bounds, max_weight, risk_free_rate)
        curr_w = w_p
        ten_portfolios.append({
            "id": pid,
            "name": pname,
            "tag": ptag,
            "group": "Yüksek Getiri Bölgesi (Sharpe Üstü)",
            "color": pcolor,
            "symbol": "triangle-up",
            "return": ret_p,
            "volatility": vol_p,
            "sharpe": sh_p,
            "weights": pd.Series(w_p, index=symbols),
        })

    max_sharpe_info = {
        "id": "MS",
        "name": "Maksimum Sharpe (Teğet Noktası)",
        "tag": "⭐ Referans Zirve",
        "color": "#10B981",
        "symbol": "star",
        "return": r_sh,
        "volatility": v_sh,
        "sharpe": s_sh,
        "weights": pd.Series(w_sh, index=symbols),
    }

    return ten_portfolios, max_sharpe_info


def generate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    points: int = 100,
    include_portfolios: List[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    TAM Markowitz Parabolünü (Hem Alt Verimsiz Kanat Hem Üst Etkin Kanat) üretir.
    - Alt Kanat: Kısıtlar altındaki en düşük getiriden Global Minimum Volatiliteye (P1) kadar.
    - Üst Kanat: Global Minimum Volatilitenin (P1) üzerinden Maksimum Getiriye (P10) kadar.
    P1..P10 ve Max Sharpe noktaları doğrudan eğri ağına dahil edilir.
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    init_weights = np.array([1.0 / n_assets] * n_assets)
    sum_c = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    # 1. Global Min Vol (P1 / parabolün burun noktası)
    res_min = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_min = np.clip(res_min.x if res_min.success else init_weights, 0, max_weight)
    w_min /= np.sum(w_min)
    r_min = float(np.dot(w_min, mu))

    # 2. Kısıtlar altında Mümkün Olan En Düşük Getiri (Alt kanadın başlangıcı)
    def pos_ret_obj(w):
        return np.dot(w, mu)

    res_lowest = minimize(pos_ret_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_lowest = np.clip(res_lowest.x if res_lowest.success else init_weights, 0, max_weight)
    w_lowest /= np.sum(w_lowest)
    r_lowest = float(np.dot(w_lowest, mu))

    # 3. Kısıtlar altında Mümkün Olan En Yüksek Getiri (Üst kanadın zirvesi)
    def neg_ret_obj(w):
        return -np.dot(w, mu)

    res_highest = minimize(neg_ret_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_c)
    w_highest = np.clip(res_highest.x if res_highest.success else init_weights, 0, max_weight)
    w_highest /= np.sum(w_highest)
    r_highest = float(np.dot(w_highest, mu))

    # Aralık güvenlik kontrolü
    if r_lowest >= r_min:
        r_lowest = r_min - 0.05
    if r_highest <= r_min:
        r_highest = r_min + 0.05

    # TAM PARABOL IZGARASI:
    # Alt Kanat: r_lowest'ten r_min'e kadar olan dönüş (yaklaşık 40 nokta)
    lower_grid = list(np.linspace(r_lowest, r_min, 40)[:-1])
    # Üst Kanat: r_min'den r_highest'a kadar olan etkin dal (yaklaşık 90 nokta)
    upper_grid = list(np.linspace(r_min, r_highest, max(80, points)))

    special_targets = []
    if include_portfolios:
        special_targets = [float(p["return"]) for p in include_portfolios]

    # Tüm ızgarayı birleştir ve getiriye göre sırala (parabolün en altından en üstüne)
    all_targets = sorted(list(set(lower_grid + upper_grid + special_targets)))

    frontier_returns = []
    frontier_volatilities = []
    frontier_sharpes = []
    frontier_weights = []

    # Alt kanat başlangıcından başlayarak sırasıyla warm-start ile çöz
    curr_w = w_lowest
    for t_r in all_targets:
        w_pt, r_pt, v_pt, s_pt = _solve_frontier_point(t_r, curr_w, mu, sigma, bounds, max_weight, risk_free_rate)
        curr_w = w_pt
        frontier_returns.append(r_pt)
        frontier_volatilities.append(v_pt)
        frontier_sharpes.append(s_pt)
        frontier_weights.append(dict(zip(symbols, w_pt)))

    return pd.DataFrame({
        "Beklenen Getiri": frontier_returns,
        "Volatilite": frontier_volatilities,
        "Sharpe": frontier_sharpes,
        "Ağırlıklar": frontier_weights,
    })


def simulate_monte_carlo(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    num_portfolios: int = 2000,
) -> pd.DataFrame:
    """Monte Carlo yöntemi ile rastgele portföyler üretir."""
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values

    results = np.zeros((3, num_portfolios))
    np.random.seed(42)
    for i in range(num_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        if np.max(w) > max_weight:
            w = np.clip(w, 0, max_weight)
            w /= np.sum(w)

        ret, vol, sh = portfolio_performance(w, mu, sigma, risk_free_rate)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sh

    df = pd.DataFrame({
        "Beklenen Getiri": results[0],
        "Volatilite": results[1],
        "Sharpe": results[2],
    })
    return df
