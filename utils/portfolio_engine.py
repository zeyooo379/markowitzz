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
