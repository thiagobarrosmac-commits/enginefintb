from fastapi import FastAPI, HTTPException
import numpy as np

from app.schemas import (
    AnalysisRequest, ReturnsResponse, SharpeResponse, MarkowitzResponse,
    CAPMResponse, CorrResponse, RiskResponse
)
from app.core import (
    fetch_adj_close, compute_daily_returns, annualize_stats, sharpe_annual,
    default_benchmark, normalize_weights, portfolio_returns,
    markowitz_simulation, solve_max_sharpe, solve_min_variance, efficient_envelope,
    capm_portfolio, corr_cov_annual,
    drawdown_series, max_drawdown, rolling_vol, rolling_sharpe
)

app = FastAPI(title="Ações Analytics API", version="1.3.0")


@app.get("/")
def root():
    return {"status": "ok", "service": "acoes-analytics-api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/returns", response_model=ReturnsResponse)
def returns(req: AnalysisRequest):
    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, ret_log = compute_daily_returns(prices)

    annual = annualize_stats(ret_simple, trading_days=req.trading_days, prices=prices)

    return {
        "daily_returns": {
            "simple": ret_simple.tail(500).to_dict(),
            "log": ret_log.tail(500).to_dict()
        },
        "annual_returns": annual.to_dict()
    }


@app.post("/sharpe", response_model=SharpeResponse)
def sharpe(req: AnalysisRequest):
    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, _ = compute_daily_returns(prices)

    sharpe_df, stats = sharpe_annual(ret_simple, rf_annual=req.rf_annual, trading_days=req.trading_days)

    return {
        "sharpe_annual": sharpe_df.to_dict(),
        "stats_annual": stats.to_dict()
    }


@app.post("/corr", response_model=CorrResponse)
def corr(req: AnalysisRequest):
    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, _ = compute_daily_returns(prices)
    corr_m, cov_a = corr_cov_annual(ret_simple, trading_days=req.trading_days)
    return {"corr": corr_m.to_dict(), "cov_annual": cov_a.to_dict()}


@app.post("/risk", response_model=RiskResponse)
def risk(req: AnalysisRequest):
    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, _ = compute_daily_returns(prices)

    dd = {}
    mdd = {}
    rv21 = {}
    rv63 = {}
    rs63 = {}

    for c in ret_simple.columns:
        s = ret_simple[c].dropna()
        dd[c] = drawdown_series(s).tail(800).to_dict()
        mdd[c] = max_drawdown(s)
        rv21[c] = rolling_vol(s, 21, trading_days=req.trading_days).tail(800).to_dict()
        rv63[c] = rolling_vol(s, 63, trading_days=req.trading_days).tail(800).to_dict()
        rs63[c] = rolling_sharpe(s, 63, rf_annual=req.rf_annual, trading_days=req.trading_days).tail(800).to_dict()

    return {
        "drawdown": dd,
        "max_drawdown": mdd,
        "rolling_vol_21": rv21,
        "rolling_vol_63": rv63,
        "rolling_sharpe_63": rs63
    }


@app.post("/markowitz", response_model=MarkowitzResponse)
def markowitz(req: AnalysisRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Markowitz requer 2+ tickers.")

    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, _ = compute_daily_returns(prices)

    pts, _, mu_a, cov_a = markowitz_simulation(
        ret_simple,
        rf_annual=req.rf_annual,
        n_portfolios=req.n_portfolios,
        trading_days=req.trading_days
    )

    env = efficient_envelope(pts)

    # --------- Otimizados ---------
    w_ms, r_ms, v_ms, s_ms = solve_max_sharpe(mu_a, cov_a, req.rf_annual)

    w_mv, v_mv = solve_min_variance(cov_a)
    # ✅ NOVO: calcular ret e sharpe do min-variance (antes vinha só vol)
    r_mv = float(w_mv @ mu_a)
    s_mv = (r_mv - req.rf_annual) / v_mv if v_mv > 0 else np.nan

    # --------- Equal weight ---------
    w_eq = normalize_weights(None, len(req.tickers))
    r_eq = float(w_eq @ mu_a)
    v_eq = float(np.sqrt(w_eq @ cov_a @ w_eq))
    s_eq = (r_eq - req.rf_annual) / v_eq if v_eq > 0 else np.nan

    # ✅ NOVO: PORT do usuário calculado no BACKEND (mesmo mu/cov do Markowitz)
    w_user = normalize_weights(req.weights, len(req.tickers))
    r_user = float(w_user @ mu_a)
    v_user = float(np.sqrt(w_user @ cov_a @ w_user))
    s_user = (r_user - req.rf_annual) / v_user if v_user > 0 else np.nan

    return {
        "frontier_points": pts.to_dict(),
        "efficient_envelope": env.to_dict(),
        "equal_weight": {
            "weights": dict(zip(req.tickers, w_eq.tolist())),
            "ret": r_eq,
            "vol": v_eq,
            "sharpe": s_eq
        },
        "max_sharpe": {
            "weights": dict(zip(req.tickers, w_ms.tolist())),
            "ret": r_ms,
            "vol": v_ms,
            "sharpe": s_ms
        },
        "min_variance": {
            "weights": dict(zip(req.tickers, w_mv.tolist())),
            "ret": r_mv,          # ✅ NOVO
            "vol": v_mv,
            "sharpe": s_mv        # ✅ NOVO
        },
        "port_user": {
            "weights": dict(zip(req.tickers, w_user.tolist())),
            "ret": r_user,
            "vol": v_user,
            "sharpe": s_user
        },
        "inputs": {
            "tickers": req.tickers,
            "rf_annual": req.rf_annual,
            "trading_days": req.trading_days,
            "n_portfolios": req.n_portfolios
        }
    }


@app.post("/capm", response_model=CAPMResponse)
def capm(req: AnalysisRequest):
    bench = req.benchmark or default_benchmark(req.tickers)

    prices_assets = fetch_adj_close(req.tickers, req.start, req.end)
    prices_bench = fetch_adj_close([bench], req.start, req.end)

    ret_assets, _ = compute_daily_returns(prices_assets)
    ret_bench, _ = compute_daily_returns(prices_bench)

    w = normalize_weights(req.weights, len(req.tickers))
    port = portfolio_returns(ret_assets, w)

    alpha_a, beta, r2, reg, capm_df = capm_portfolio(
        port_daily=port,
        bench_daily=ret_bench.iloc[:, 0],
        rf_annual=req.rf_annual,
        trading_days=req.trading_days
    )

    capm_tail = capm_df.tail(800)

    return {
        "benchmark": bench,
        "alpha_annual": alpha_a,
        "beta": beta,
        "r2": r2,
        "regression": reg,
        "scatter": capm_tail.to_dict()
    }
