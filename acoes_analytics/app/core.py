"""
QuantView — Core Analytics Engine
==================================
Módulo de cálculos financeiros: retornos, risco, Markowitz, CAPM,
valuation por múltiplos, VaR e stress testing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize, stats as sp_stats
from functools import lru_cache
from typing import Optional

# =============================
# Defaults / Helpers
# =============================

def default_benchmark(tickers: list[str]) -> str:
    """Retorna benchmark padrão baseado no mercado dos tickers."""
    if all(t.endswith(".SA") for t in tickers):
        return "^BVSP"
    return "^GSPC"


def normalize_weights(weights: Optional[list[float]], n: int) -> np.ndarray:
    """Normaliza pesos para somar 1. Se None, retorna equal-weight."""
    if weights is None:
        return np.ones(n) / n
    w = np.array(weights, dtype=float)
    if len(w) != n:
        raise ValueError(f"weights deve ter {n} elementos, recebeu {len(w)}.")
    if np.isclose(w.sum(), 0):
        raise ValueError("Soma dos pesos não pode ser zero.")
    return w / w.sum()


# =============================
# Data Fetch (com cache)
# =============================

@lru_cache(maxsize=64)
def _fetch_adj_close_cached(tickers_tuple: tuple, start: str, end: str) -> pd.DataFrame:
    """Baixa preços ajustados do yfinance com cache."""
    tickers = list(tickers_tuple)

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or len(df) == 0:
        raise ValueError("Não foi possível baixar dados do yfinance. Verifique tickers/datas.")

    # Com auto_adjust=True, "Close" já é o preço ajustado
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            first_level = df.columns.get_level_values(0)[0]
            px = df[first_level].copy()
    else:
        if "Close" in df.columns:
            px = df[["Close"]].copy()
            px.columns = [tickers[0]]
        else:
            raise ValueError("yfinance não retornou 'Close'.")

    px = px.dropna(how="all").ffill()
    px = px.dropna(axis=1, how="all")
    px = px.dropna()

    if px.shape[1] == 0 or len(px) < 2:
        raise ValueError("Preços insuficientes após limpeza. Verifique tickers/datas.")

    return px


def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wrapper público que retorna cópia (protege o cache)."""
    return _fetch_adj_close_cached(tuple(tickers), start, end).copy()


# =============================
# Returns
# =============================

def compute_daily_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula retornos diários simples e log a partir dos preços."""
    prices = prices.copy()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(axis=1, how="all").ffill().dropna()

    ret_simple = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret_log = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return ret_simple, ret_log


def cagr_from_prices(prices: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """CAGR (Compound Annual Growth Rate) calculado diretamente dos preços."""
    px = prices.copy()
    px = px.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    px = px.dropna(axis=1, how="all").ffill().dropna()
    n = px.shape[0]
    if n < 2:
        return pd.Series(index=px.columns, dtype=float)
    years = n / trading_days
    return (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1


def annualize_stats(
    daily_returns: pd.DataFrame,
    trading_days: int = 252,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estatísticas anualizadas:
      ret_annual = (1 + mean_daily)^252 - 1
      vol_annual = std_daily * sqrt(252)
      cagr (opcional, baseado em preços)
    """
    dr = daily_returns.copy()
    dr = dr.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dr = dr.dropna(axis=1, how="all")

    if dr.empty:
        return pd.DataFrame(columns=["ret_annual", "vol_annual", "cagr"])

    mu_daily = dr.mean(skipna=True)
    vol_daily = dr.std(ddof=1, skipna=True)

    ret_annual = (1 + mu_daily) ** trading_days - 1
    vol_annual = vol_daily * np.sqrt(trading_days)

    out = pd.DataFrame({"ret_annual": ret_annual, "vol_annual": vol_annual})
    out = out.replace([np.inf, -np.inf], np.nan)

    if prices is not None:
        out["cagr"] = cagr_from_prices(prices, trading_days=trading_days)

    return out


def sharpe_annual(
    daily_returns: pd.DataFrame,
    rf_annual: float = 0.10,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula Sharpe ratio anualizado para cada ativo."""
    stats = annualize_stats(daily_returns, trading_days=trading_days)
    if stats.empty:
        return pd.DataFrame(columns=["sharpe_annual"]), stats

    stats2 = stats.copy()
    stats2 = stats2.replace([np.inf, -np.inf], np.nan)
    stats2 = stats2.dropna(subset=["vol_annual"], how="any")

    sharpe = (stats2["ret_annual"] - rf_annual) / stats2["vol_annual"]
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    return sharpe.to_frame("sharpe_annual"), stats2


# =============================
# Portfolio
# =============================

def portfolio_returns(daily_returns: pd.DataFrame, weights) -> pd.Series:
    """Retorno diário do portfólio (combinação linear dos ativos)."""
    w = np.array(weights, dtype=float)
    pr = daily_returns.values @ w
    return pd.Series(pr, index=daily_returns.index, name="portfolio")


# =============================
# Markowitz
# =============================

def markowitz_simulation(
    daily_returns: pd.DataFrame,
    rf_annual: float = 0.10,
    n_portfolios: int = 8000,
    trading_days: int = 252,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Simulação de Monte Carlo para fronteira de Markowitz."""
    rng = np.random.default_rng(seed)
    n = daily_returns.shape[1]

    mu_d = daily_returns.mean().values
    cov_d = daily_returns.cov().values

    mu_a = (1 + mu_d) ** trading_days - 1
    cov_a = cov_d * trading_days

    Ws, rets, vols, sharpes = [], [], [], []
    for _ in range(int(n_portfolios)):
        w = rng.random(n)
        w = w / w.sum()
        r = float(w @ mu_a)
        v = float(np.sqrt(w @ cov_a @ w))
        s = (r - rf_annual) / v if v > 0 else np.nan
        Ws.append(w)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)

    df = pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpes})
    W = np.vstack(Ws)
    return df, W, mu_a, cov_a


def solve_max_sharpe(
    mu_a: np.ndarray, cov_a: np.ndarray, rf_annual: float
) -> tuple[np.ndarray, float, float, float]:
    """Otimização: portfólio de máximo Sharpe (tangency portfolio)."""
    n = len(mu_a)

    def neg_sharpe(w):
        r = w @ mu_a
        v = np.sqrt(w @ cov_a @ w)
        return -((r - rf_annual) / v) if v > 0 else 0

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = optimize.minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = res.x
    r = float(w @ mu_a)
    v = float(np.sqrt(w @ cov_a @ w))
    s = (r - rf_annual) / v if v > 0 else np.nan
    return w, r, v, s


def solve_min_variance(cov_a: np.ndarray) -> tuple[np.ndarray, float]:
    """Otimização: portfólio de mínima variância."""
    n = cov_a.shape[0]

    def port_var(w):
        return w @ cov_a @ w

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = optimize.minimize(port_var, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = res.x
    v = float(np.sqrt(w @ cov_a @ w))
    return w, v


def efficient_envelope(points_df: pd.DataFrame) -> pd.DataFrame:
    """Extrai o envelope eficiente (upper boundary) da nuvem de Markowitz."""
    df = points_df.dropna(subset=["vol", "ret"]).sort_values("vol").copy()
    max_ret = -np.inf
    keep = []
    for _, row in df.iterrows():
        if row["ret"] >= max_ret:
            keep.append(True)
            max_ret = row["ret"]
        else:
            keep.append(False)
    env = df.loc[keep, ["vol", "ret"]].drop_duplicates().sort_values("vol")
    return env


# =============================
# Correlação / Covariância
# =============================

def corr_cov_annual(
    daily_returns: pd.DataFrame, trading_days: int = 252
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz de correlação e covariância anualizada."""
    corr = daily_returns.corr()
    cov_annual = daily_returns.cov() * trading_days
    return corr, cov_annual


# =============================
# Risk Metrics
# =============================

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Série temporal de drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum / peak) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (valor mais negativo do drawdown)."""
    return float(drawdown_series(returns).min())


def rolling_vol(returns: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    """Volatilidade rolling anualizada."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    rf_annual: float = 0.10,
    trading_days: int = 252,
) -> pd.Series:
    """Sharpe ratio rolling anualizado."""
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    ex = returns - rf_daily
    mu = ex.rolling(window).mean() * trading_days
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    return mu / vol


# =============================
# VaR (Value at Risk)
# =============================

def var_parametric(
    daily_returns: pd.Series,
    confidence: float = 0.95,
    trading_days: int = 252,
) -> dict:
    """
    VaR paramétrico (Gaussiano), histórico e CVaR (Expected Shortfall).
    Retorna perda diária e anualizada para o nível de confiança dado.
    """
    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std(ddof=1))
    z = sp_stats.norm.ppf(1 - confidence)

    var_daily_param = -(mu + z * sigma)
    var_annual_param = var_daily_param * np.sqrt(trading_days)

    # VaR histórico
    var_daily_hist = -float(daily_returns.quantile(1 - confidence))

    # CVaR (Expected Shortfall)
    tail = daily_returns[daily_returns <= daily_returns.quantile(1 - confidence)]
    cvar_daily = -float(tail.mean()) if len(tail) > 0 else np.nan

    return {
        "confidence": confidence,
        "var_daily_parametric": round(var_daily_param, 6),
        "var_annual_parametric": round(var_annual_param, 6),
        "var_daily_historic": round(var_daily_hist, 6),
        "cvar_daily": round(cvar_daily, 6),
    }


# =============================
# Stress Testing
# =============================

def stress_scenarios(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    trading_days: int = 252,
) -> list[dict]:
    """
    Cenários de stress: Bear (-2σ), Base (esperado), Bull (+2σ).
    Retorna retorno anualizado do portfólio em cada cenário.
    """
    port_ret = daily_returns.values @ weights
    mu_daily = float(np.mean(port_ret))
    sigma_daily = float(np.std(port_ret, ddof=1))

    scenarios = []
    for label, shift in [("Bear (-2σ)", -2), ("Base", 0), ("Bull (+2σ)", 2)]:
        daily_scenario = mu_daily + shift * sigma_daily
        annual_ret = (1 + daily_scenario) ** trading_days - 1
        scenarios.append({
            "scenario": label,
            "daily_return": round(daily_scenario, 6),
            "annual_return": round(annual_ret, 4),
        })
    return scenarios


# =============================
# Valuation Multiples
# =============================

def fetch_valuation_multiples(tickers: list[str]) -> pd.DataFrame:
    """
    Busca múltiplos de valuation via yfinance para cada ticker.
    Retorna DataFrame com EV/EBITDA, P/E, EV/Revenue, P/B, etc.
    """
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            rows.append({
                "ticker": t,
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "ev_revenue": info.get("enterpriseToRevenue"),
                "pb": info.get("priceToBook"),
                "ps": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            })
        except Exception:
            rows.append({"ticker": t})

    df = pd.DataFrame(rows).set_index("ticker")
    return df


# =============================
# CAPM
# =============================

def capm_portfolio(
    port_daily: pd.Series,
    bench_daily: pd.Series,
    rf_annual: float = 0.10,
    trading_days: int = 252,
) -> tuple[float, float, float, dict, pd.DataFrame]:
    """
    Regressão CAPM do portfólio contra benchmark.
    Retorna: alpha_annual, beta, R², dict de regressão, DataFrame scatter.
    """
    df = pd.concat([port_daily, bench_daily], axis=1).dropna()
    df.columns = ["port", "bench"]

    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    y = (df["port"] - rf_daily).values
    x = (df["bench"] - rf_daily).values

    x_mean = x.mean()
    y_mean = y.mean()
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.mean((x - x_mean) ** 2)

    beta = cov_xy / var_x if var_x > 0 else np.nan
    alpha_daily = y_mean - beta * x_mean

    y_hat = alpha_daily + beta * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_annual = (1 + alpha_daily) ** trading_days - 1

    capm_df = pd.DataFrame({"x": x, "y": y, "y_hat": y_hat}, index=df.index)
    reg = {
        "alpha_daily": float(alpha_daily),
        "rf_daily": float(rf_daily),
        "n_obs": int(len(df)),
    }

    return float(alpha_annual), float(beta), float(r2), reg, capm_df
