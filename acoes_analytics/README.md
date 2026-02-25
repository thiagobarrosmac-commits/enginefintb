# 📊 QuantView — Quantitative Portfolio Analytics

Full-stack platform for quantitative equity analysis, combining Modern Portfolio Theory (MPT), CAPM regression, risk analytics, and peer valuation. Built with a **FastAPI** backend and **Streamlit** dashboard.

---

## Features

| Module | Description |
|--------|-------------|
| **Returns** | Daily (simple & log) and annualized returns with CAGR |
| **Sharpe Ratio** | Annualized Sharpe with risk-return scatter |
| **Correlation** | Correlation and covariance matrices with heatmaps |
| **Risk Metrics** | Drawdown, max drawdown, rolling volatility (21/63d), rolling Sharpe |
| **Markowitz** | Monte Carlo simulation, efficient frontier, max Sharpe & min variance optimization |
| **CAPM** | Alpha, beta, R² regression with multi-benchmark comparison |
| **VaR & Stress** | Value at Risk (parametric & historic, 95/99%), CVaR, bear/base/bull scenarios |
| **Valuation** | Peer comparison: EV/EBITDA, P/E, EV/Revenue, P/B, margins, ROE, debt ratios |

## Architecture

```
┌──────────────────┐       REST/JSON       ┌──────────────────┐
│   Streamlit UI   │ ◄──────────────────► │   FastAPI API    │
│  (Streamlit Cloud)│                      │    (Render)      │
└──────────────────┘                      └──────────────────┘
                                                   │
                                            ┌──────┴──────┐
                                            │  yfinance   │
                                            │  (Yahoo)    │
                                            └─────────────┘
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn app.api:app --reload --port 8000
```

### 3. Run the Dashboard
```bash
# In another terminal
streamlit run streamlit_app.py
```

The dashboard expects the API URL in the `API_URL` environment variable (defaults to `http://127.0.0.1:8000`).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `POST` | `/returns` | Daily & annualized returns |
| `POST` | `/sharpe` | Sharpe ratio |
| `POST` | `/corr` | Correlation & covariance |
| `POST` | `/risk` | Drawdown & rolling metrics |
| `POST` | `/markowitz` | Portfolio optimization |
| `POST` | `/capm` | CAPM regression |
| `POST` | `/var` | VaR, CVaR & stress testing |
| `POST` | `/valuation` | Valuation multiples |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rf_annual` | 0.10 | Risk-free rate (annualized) |
| `trading_days` | 252 | Trading days per year |
| `n_portfolios` | 8,000 | Monte Carlo simulations |

## Tech Stack

- **Backend**: Python, FastAPI, Pydantic
- **Frontend**: Streamlit, Plotly
- **Data**: yfinance, pandas, numpy, scipy
- **Deployment**: Render (API) + Streamlit Cloud (UI)

## Roadmap

- [ ] DCF (Discounted Cash Flow) valuation model
- [ ] Monte Carlo simulation for portfolio projections
- [ ] Sector screening and filtering
- [ ] PDF report export
- [ ] Authentication and saved portfolios

---

*Built for quantitative analysis and portfolio management research.*
