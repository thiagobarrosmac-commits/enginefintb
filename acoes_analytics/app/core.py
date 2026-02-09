import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize
from functools import lru_cache

# =============================
# Defaults / Helpers
# =============================
def default_benchmark(tickers):
    if all(t.endswith(".SA") for t in tickers):
        return "^BVSP"
    return "^GSPC"

def normalize_weights(weights, n):
    if weights is None:
        return np.ones(n) / n
    w = np.array(weights, dtype=float)
    if len(w) != n:
        raise ValueError("weights deve ter o mesmo tamanho de tickers.")
    if np.isclose(w.sum(), 0):
        raise ValueError("weights soma zero.")
    w = w / w.sum()
    return w

# =============================
# Data fetch (cache)
# =============================
@lru_cache(maxsize=64)
def _fetch_adj_close_cached(tickers_tuple, start, end):
    tickers = list(tickers_tuple)

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True
    )
    if df is None or len(df) == 0:
        raise ValueError("Não foi possível baixar dados do yfinance. Verifique tickers/datas.")

    # MultiIndex comum: ('Adj Close', 'JPM'), ('Adj Close', 'MA')...
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            px = df["Adj Close"].copy()
        elif "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            raise ValueError("yfinance não retornou 'Adj Close' nem 'Close'.")
    else:
        # caso 1 ticker
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].copy()
            px.columns = [tickers[0]]
        elif "Close" in df.columns:
            px = df[["Close"]].copy()
            px.columns = [tickers[0]]
        else:
            raise ValueError("yfinance não retornou 'Adj Close' nem 'Close' (single).")

    # limpeza
    px = px.dropna(how="all").ffill().dropna()
    # remove colunas 100% NaN
    px = px.dropna(axis=1, how="all")

    if px.shape[1] == 0:
        raise ValueError("Preços vazios após limpeza (tickers/datas inválidos).")

    return px

def fetch_adj_close(tickers, start, end):
    return _fetch_adj_close_cached(tuple(tickers), start, end).copy()

# =============================
# 1) Daily returns
# =============================
def compute_daily_returns(prices: pd.DataFrame):
    prices = prices.copy().astype(float)
    prices = prices.dropna(axis=1, how="all").ffill().dropna()

    ret_simple = prices.pct_change().dropna()
    ret_log = np.log(prices / prices.shift(1)).dropna()
    return ret_simple, ret_log

# =============================
# 2) Annualization
# =============================
def annualize_stats(daily_returns: pd.DataFrame, trading_days=252):
    """
    Robust: remove colunas vazias / infinities para evitar ret_annual virar NaN.
    """
    dr = daily_returns.copy()

    # garante numérico e remove colunas que ficaram vazias
    dr = dr.apply(pd.to_numeric, errors="coerce")
    dr = dr.replace([np.inf, -np.inf], np.nan)
    dr = dr.dropna(axis=1, how="all")
    dr = dr.dropna()

    if dr.empty:
        # devolve DF vazio com colunas corretas
        return pd.DataFrame(columns=["ret_annual", "vol_annual"])

    mu_daily = dr.mean()
    vol_daily = dr.std(ddof=1)

    # retorno anualizado por média diária (simples)
    mu_annual = (1 + mu_daily) ** trading_days - 1
    vol_annual = vol_daily * np.sqrt(trading_days)

    out = pd.DataFrame({"ret_annual": mu_annual, "vol_annual": vol_annual})
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

# =============================
# 3) Sharpe annual
# =============================
def sharpe_annual(daily_returns: pd.DataFrame, rf_annual=0.10, trading_days=252):
    stats = annualize_stats(daily_returns, trading_days=trading_days)

    if stats.empty:
        sharpe = pd.Series(dtype=float, name="sharpe_annual")
        return sharpe.to_frame("sharpe_annual"), stats

    sharpe = (stats["ret_annual"] - rf_annual) / stats["vol_annual"]
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe.to_frame("sharpe_annual"), stats

# =============================
# Portfolio returns / stats
# =============================
def portfolio_returns(daily_returns: pd.DataFrame, weights):
    w = np.array(weights, dtype=float)
    pr = daily_returns.values @ w
    return pd.Series(pr, index=daily_returns.index, name="portfolio")

def portfolio_stats(daily_returns: pd.DataFrame, weights, trading_days=252):
    pr = portfolio_returns(daily_returns, weights)
    mu_d = pr.mean()
    vol_d = pr.std(ddof=1)
    mu_a = (1 + mu_d) ** trading_days - 1
    vol_a = vol_d * np.sqrt(trading_days)
    return float(mu_a), float(vol_a), pr

# =============================
# 4) Markowitz
# =============================
def markowitz_simulation(
    daily_returns: pd.DataFrame,
    rf_annual=0.10,
    n_portfolios=8000,
    trading_days=252,
    seed=42
):
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

def solve_max_sharpe(mu_a, cov_a, rf_annual):
    n = len(mu_a)

    def neg_sharpe(w):
        w = np.array(w)
        r = w @ mu_a
        v = np.sqrt(w @ cov_a @ w)
        return -((r - rf_annual) / v)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = optimize.minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = res.x
    r = float(w @ mu_a)
    v = float(np.sqrt(w @ cov_a @ w))
    s = (r - rf_annual) / v if v > 0 else np.nan
    return w, r, v, s

def solve_min_variance(cov_a):
    n = cov_a.shape[0]

    def port_var(w):
        w = np.array(w)
        return w @ cov_a @ w

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = optimize.minimize(port_var, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = res.x
    v = float(np.sqrt(w @ cov_a @ w))
    return w, v

def efficient_envelope(points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Envelope superior (fronteira eficiente aproximada) a partir de pontos simulados.
    Mantém recordes de retorno ao ordenar por vol.
    """
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
# Corr / Cov annual
# =============================
def corr_cov_annual(daily_returns: pd.DataFrame, trading_days=252):
    corr = daily_returns.corr()
    cov_annual = daily_returns.cov() * trading_days
    return corr, cov_annual

# =============================
# Risk extras
# =============================
def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd

def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())

def rolling_vol(returns: pd.Series, window: int, trading_days=252) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)

def rolling_sharpe(returns: pd.Series, window: int, rf_annual=0.10, trading_days=252) -> pd.Series:
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    ex = returns - rf_daily
    mu = ex.rolling(window).mean() * trading_days
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    return mu / vol

# =============================
# 5) CAPM (5 retornos: alpha_a, beta, r2, reg_dict, capm_df)
# =============================
def capm_portfolio(port_daily: pd.Series, bench_daily: pd.Series, rf_annual=0.10, trading_days=252):
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
        "n_obs": int(len(df))
    }
    return float(alpha_annual), float(beta), float(r2), reg, capm_df
