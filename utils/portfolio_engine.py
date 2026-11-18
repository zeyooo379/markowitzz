"""
Markowitz Portföy Optimizasyon Motoru
Dönemsel (Kullanıcının seçtiği T gün) ve Yıllıklaştırılmış (252 gün) hesaplamaları destekler.
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
    mode: str = "period",  # 'period' veya 'annualized'
    annual_trading_days: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Kovaryans, getiri ve risk parametrelerini seçilen ölçeğe göre hesaplar.
    
    - 'period': Kullanıcının seçtiği tam gün sayısı (T) baz alınır.
      Örneğin 3 ay (63 gün) veri çekildiyse, getiri ve kovaryans 63 gün üzerinden hesaplanır.
    - 'annualized': Standart 252 güne projeksiyon yapılır.
    """
    actual_days = len(returns)
    if actual_days == 0:
        raise ValueError("Getiri serisi boş olamaz.")

    daily_mean = returns.mean()
    daily_cov = returns.cov()
    daily_std = returns.std()
    correlation = returns.corr()

    # Kümülatif gerçekleşen dönem getirisi (fiyat başlangıçtan bitişe)
    if not prices.empty:
        cumulative_period_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    else:
        cumulative_period_return = daily_mean * actual_days

    if mode == "period":
        time_factor = float(actual_days)
        scale_label = f"Seçilen Dönem ({actual_days} İşlem Günü)"
        # Risksiz faiz oranını döneme indirge (bileşik formül)
        period_rf = (1 + annual_risk_free_rate) ** (actual_days / annual_trading_days) - 1
    else:
        time_factor = float(annual_trading_days)
        scale_label = f"Yıllıklaştırılmış ({annual_trading_days} İşlem Günü Projeksiyonu)"
        period_rf = annual_risk_free_rate

    # Ortalama beklenen getiri ve kovaryans
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

    res_min_vol = minimize(
        min_vol_obj,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_min_vol = res_min_vol.x if res_min_vol.success else init_weights
    w_min_vol = np.clip(w_min_vol, 0, max_weight)
    w_min_vol /= np.sum(w_min_vol)
    ret_min_vol, vol_min_vol, sh_min_vol = portfolio_performance(w_min_vol, mu, sigma, risk_free_rate)

    # 2. Maksimum Sharpe Portföyü
    def neg_sharpe_obj(w):
        r, v, s = portfolio_performance(w, mu, sigma, risk_free_rate)
        return -s

    res_max_sh = minimize(
        neg_sharpe_obj,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_max_sh = res_max_sh.x if res_max_sh.success else init_weights
    w_max_sh = np.clip(w_max_sh, 0, max_weight)
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


def generate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    points: int = 50,
) -> pd.DataFrame:
    """
    Etkin Sınır (Efficient Frontier) eğrisi için noktaları ve 
    her noktaya karşılık gelen optimal hisse ağırlıklarını hesaplar.
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))

    min_ret = float(np.min(mu))
    max_ret = float(np.max(mu))
    target_returns = np.linspace(min_ret, max_ret, points)

    frontier_returns = []
    frontier_volatilities = []
    frontier_sharpes = []
    frontier_weights = []

    init_weights = np.array([1.0 / n_assets] * n_assets)

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, mu) - target},
        )
        def vol_obj(w):
            return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

        res = minimize(vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            w = np.clip(res.x, 0, max_weight)
            w /= np.sum(w)
            ret, vol, sh = portfolio_performance(w, mu, sigma, risk_free_rate)
            frontier_returns.append(ret)
            frontier_volatilities.append(vol)
            frontier_sharpes.append(sh)
            frontier_weights.append(dict(zip(symbols, w)))

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


def get_five_frontier_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Etkin Sınır (Efficient Frontier) üzerinde 5 stratejik portföy hesaplar:
    - P1: Çok Düşük Risk (Defansif / Min Volatilite)
    - P2: Temkinli Büyüme (Düşük-Orta Risk)
    - P3: Optimal Denge (Maksimum Sharpe / Teğet Portföy)
    - P4: Dinamik Büyüme (Orta-Yüksek Risk)
    - P5: Agresif (Maksimum Getiri)
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    init_weights = np.array([1.0 / n_assets] * n_assets)
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # 1. P1: Minimum Volatilite
    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))
    
    res_min_vol = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_min = np.clip(res_min_vol.x, 0, max_weight)
    w_min /= np.sum(w_min)
    r_min, v_min, s_min = portfolio_performance(w_min, mu, sigma, risk_free_rate)

    # 3. P3: Maksimum Sharpe
    def neg_sharpe_obj(w):
        r, v, s = portfolio_performance(w, mu, sigma, risk_free_rate)
        return -s

    res_max_sh = minimize(neg_sharpe_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_sh = np.clip(res_max_sh.x, 0, max_weight)
    w_sh /= np.sum(w_sh)
    r_sh, v_sh, s_sh = portfolio_performance(w_sh, mu, sigma, risk_free_rate)

    # 5. P5: Maksimum Getiri
    def neg_ret_obj(w):
        return -np.dot(w, mu)

    res_max_ret = minimize(neg_ret_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_max_ret = np.clip(res_max_ret.x, 0, max_weight)
    w_max_ret /= np.sum(w_max_ret)
    r_max, v_max, s_max = portfolio_performance(w_max_ret, mu, sigma, risk_free_rate)

    # 2. P2: Temkinli Büyüme (Hedef Getiri: Min Vol ile Max Sharpe arası)
    target_r2 = (r_min + r_sh) / 2.0
    c2 = (
        sum_constraint,
        {"type": "eq", "fun": lambda w, t=target_r2: np.dot(w, mu) - t},
    )
    res_p2 = minimize(min_vol_obj, (w_min + w_sh) / 2.0, method="SLSQP", bounds=bounds, constraints=c2)
    w_p2 = np.clip(res_p2.x if res_p2.success else (w_min + w_sh) / 2.0, 0, max_weight)
    w_p2 /= np.sum(w_p2)
    r_p2, v_p2, s_p2 = portfolio_performance(w_p2, mu, sigma, risk_free_rate)

    # 4. P4: Dinamik Büyüme (Hedef Getiri: Max Sharpe ile Max Getiri arası)
    target_r4 = (r_sh + r_max) / 2.0
    c4 = (
        sum_constraint,
        {"type": "eq", "fun": lambda w, t=target_r4: np.dot(w, mu) - t},
    )
    res_p4 = minimize(min_vol_obj, (w_sh + w_max_ret) / 2.0, method="SLSQP", bounds=bounds, constraints=c4)
    w_p4 = np.clip(res_p4.x if res_p4.success else (w_sh + w_max_ret) / 2.0, 0, max_weight)
    w_p4 /= np.sum(w_p4)
    r_p4, v_p4, s_p4 = portfolio_performance(w_p4, mu, sigma, risk_free_rate)

    portfolios = [
        {
            "id": "P1",
            "name": "P1: Çok Düşük Risk (Defansif)",
            "short_name": "P1: Defansif",
            "tag": "🛡️ En Güvenli Liman",
            "color": "#06B6D4",
            "symbol": "diamond",
            "return": r_min,
            "volatility": v_min,
            "sharpe": s_min,
            "weights": pd.Series(w_min, index=symbols),
            "description": "Portföy dalgalanmasını minimuma indiren, sermayeyi korumayı hedefleyen defansif sepet.",
        },
        {
            "id": "P2",
            "name": "P2: Temkinli Büyüme",
            "short_name": "P2: Temkinli",
            "tag": "🌱 Düşük-Orta Risk",
            "color": "#14B8A6",
            "symbol": "hexagon",
            "return": r_p2,
            "volatility": v_p2,
            "sharpe": s_p2,
            "weights": pd.Series(w_p2, index=symbols),
            "description": "Düşük risk seviyesini korurken düzenli getiri artışı hedefleyen muhafazakar büyüme sepeti.",
        },
        {
            "id": "P3",
            "name": "P3: Optimal Denge (Maksimum Sharpe)",
            "short_name": "P3: Max Sharpe",
            "tag": "⭐ Matematiksel Zirve (Teğet)",
            "color": "#10B981",
            "symbol": "star",
            "return": r_sh,
            "volatility": v_sh,
            "sharpe": s_sh,
            "weights": pd.Series(w_sh, index=symbols),
            "description": "Risk başına en yüksek birim getiriyi sunan, piyasadaki en akılcı ve dengeli portföy.",
        },
        {
            "id": "P4",
            "name": "P4: Dinamik Büyüme",
            "short_name": "P4: Büyüme",
            "tag": "🚀 Yüksek Getiri Potansiyeli",
            "color": "#F59E0B",
            "symbol": "cross",
            "return": r_p4,
            "volatility": v_p4,
            "sharpe": s_p4,
            "weights": pd.Series(w_p4, index=symbols),
            "description": "Bir miktar daha fazla riski göze alarak getirisini teğet portföyün üzerine çıkarmak isteyenler için.",
        },
        {
            "id": "P5",
            "name": "P5: Agresif (Maksimum Getiri)",
            "short_name": "P5: Agresif",
            "tag": "🔥 Maksimum Kazanç Odaklı",
            "color": "#EF4444",
            "symbol": "triangle-up",
            "return": r_max,
            "volatility": v_max,
            "sharpe": s_max,
            "weights": pd.Series(w_max_ret, index=symbols),
            "description": "Risk kısıtı gözetmeksizin seçilen hisseler içinde en yüksek getiriyi hedefleyen agresif sepet.",
        },
    ]
    return portfolios


def get_ten_frontier_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Etkin Sınır (Efficient Frontier) üzerinde tam 10 stratejik portföy hesaplar:
    - 5 Tanesi Maksimum Sharpe'ın Altında (Düşük Risk / Defansif Bölge: P1 - P5)
    - 5 Tanesi Maksimum Sharpe'ın Üstünde (Yüksek Getiri / Dinamik Bölge: P6 - P10)
    Ayrıca merkez referans olarak Maksimum Sharpe portföyünü de döndürür.
    """
    n_assets = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values
    symbols = list(expected_returns.index)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    init_weights = np.array([1.0 / n_assets] * n_assets)
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def min_vol_obj(w):
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    # 1. Min Volatility
    res_min_vol = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_min = np.clip(res_min_vol.x if res_min_vol.success else init_weights, 0, max_weight)
    w_min /= np.sum(w_min)
    r_min, v_min, s_min = portfolio_performance(w_min, mu, sigma, risk_free_rate)

    # 2. Max Sharpe
    def neg_sharpe_obj(w):
        r, v, s = portfolio_performance(w, mu, sigma, risk_free_rate)
        return -s
    res_max_sh = minimize(neg_sharpe_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_sh = np.clip(res_max_sh.x if res_max_sh.success else init_weights, 0, max_weight)
    w_sh /= np.sum(w_sh)
    r_sh, v_sh, s_sh = portfolio_performance(w_sh, mu, sigma, risk_free_rate)

    # 3. Max Return
    def neg_ret_obj(w):
        return -np.dot(w, mu)
    res_max_ret = minimize(neg_ret_obj, init_weights, method="SLSQP", bounds=bounds, constraints=sum_constraint)
    w_max = np.clip(res_max_ret.x if res_max_ret.success else init_weights, 0, max_weight)
    w_max /= np.sum(w_max)
    r_max, v_max, s_max = portfolio_performance(w_max, mu, sigma, risk_free_rate)

    # Güvenlik kontrolü
    if r_sh <= r_min:
        r_sh = r_min + (r_max - r_min) * 0.4
    if r_max <= r_sh:
        r_max = r_sh * 1.2

    # 5 Nokta Sharpe Altında (r_min'den r_sh'ye 5 eşit aralık)
    targets_below = np.linspace(r_min, r_sh, 6)[:-1]  # 5 nokta
    # 5 Nokta Sharpe Üstünde (r_sh'den r_max'a 5 eşit aralık)
    targets_above = np.linspace(r_sh, r_max, 6)[1:]   # 5 nokta

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

    # Sharpe Altındaki 5 Portföyü Çöz
    for (pid, pname, ptag, pcolor), target_r in zip(below_names, targets_below):
        c = (sum_constraint, {"type": "eq", "fun": lambda w, t=target_r: np.dot(w, mu) - t})
        res = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=c)
        w = np.clip(res.x if res.success else init_weights, 0, max_weight)
        w /= np.sum(w)
        ret, vol, sh = portfolio_performance(w, mu, sigma, risk_free_rate)
        ten_portfolios.append({
            "id": pid,
            "name": pname,
            "tag": ptag,
            "group": "Düşük Risk Bölgesi (Sharpe Altı)",
            "color": pcolor,
            "symbol": "diamond",
            "return": ret,
            "volatility": vol,
            "sharpe": sh,
            "weights": pd.Series(w, index=symbols),
        })

    # Sharpe Üstündeki 5 Portföyü Çöz
    for (pid, pname, ptag, pcolor), target_r in zip(above_names, targets_above):
        c = (sum_constraint, {"type": "eq", "fun": lambda w, t=target_r: np.dot(w, mu) - t})
        res = minimize(min_vol_obj, init_weights, method="SLSQP", bounds=bounds, constraints=c)
        w = np.clip(res.x if res.success else init_weights, 0, max_weight)
        w /= np.sum(w)
        ret, vol, sh = portfolio_performance(w, mu, sigma, risk_free_rate)
        ten_portfolios.append({
            "id": pid,
            "name": pname,
            "tag": ptag,
            "group": "Yüksek Getiri Bölgesi (Sharpe Üstü)",
            "color": pcolor,
            "symbol": "triangle-up",
            "return": ret,
            "volatility": vol,
            "sharpe": sh,
            "weights": pd.Series(w, index=symbols),
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
