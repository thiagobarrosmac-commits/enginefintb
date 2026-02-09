from fastapi import FastAPI, HTTPException
from app.schemas import (
    AnalysisRequest, ReturnsResponse, SharpeResponse, MarkowitzResponse, CAPMResponse
)
from app.core import (
    fetch_adj_close, compute_daily_returns, annualize_stats, sharpe_annual,
    default_benchmark, normalize_weights, portfolio_returns, markowitz_simulation,
    solve_max_sharpe, solve_min_variance, capm_portfolio
)

app = FastAPI(title="Ações Analytics API", version="1.0.1")

# ✅ para não retornar 404 na raiz
@app.get("/")
def root():
    return {"status": "ok", "service": "acoes-analytics-api"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/returns", response_model=ReturnsResponse)
def returns(req: AnalysisRequest):
    _ = req.benchmark or default_benchmark(req.tickers)
    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, ret_log = compute_daily_returns(prices)

    return {
        "daily_returns": {
            "simple": ret_simple.tail(500).to_dict(),
            "log": ret_log.tail(500).to_dict()
        },
        "annual_returns": annualize_stats(ret_simple, trading_days=req.trading_days).to_dict()
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

@app.post("/markowitz", response_model=MarkowitzResponse)
def markowitz(req: AnalysisRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Markowitz requer 2+ tickers.")

    prices = fetch_adj_close(req.tickers, req.start, req.end)
    ret_simple, _ = compute_daily_returns(prices)

    df_sim, _, mu_a, cov_a = markowitz_simulation(
        ret_simple,
        rf_annual=req.rf_annual,
        n_portfolios=req.n_portfolios,
        trading_days=req.trading_days
    )

    w_ms, r_ms, v_ms, s_ms = solve_max_sharpe(mu_a, cov_a, req.rf_annual)
    w_mv, v_mv = solve_min_variance(cov_a)

    return {
        "frontier_points": df_sim.to_dict(),
        "max_sharpe": {
            "weights": dict(zip(req.tickers, w_ms.tolist())),
            "ret": r_ms, "vol": v_ms, "sharpe": s_ms
        },
        "min_variance": {
            "weights": dict(zip(req.tickers, w_mv.tolist())),
            "vol": v_mv
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

    alpha_a, beta, r2, reg = capm_portfolio(
        port_daily=port,
        bench_daily=ret_bench.iloc[:, 0],
        rf_annual=req.rf_annual,
        trading_days=req.trading_days
    )

    return {
        "benchmark": bench,
        "alpha_annual": alpha_a,
        "beta": beta,
        "r2": r2,
        "regression": reg
    }
