from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisRequest(BaseModel):
    # ✅ limite de tickers (ajuste como quiser)
    tickers: List[str] = Field(..., min_length=1, max_length=10)

    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    benchmark: Optional[str] = None

    rf_annual: float = Field(0.10, ge=0.0, le=1.0)
    weights: Optional[List[float]] = None

    trading_days: int = Field(252, ge=200, le=270)

    # ✅ limite de simulação (ajuste)
    n_portfolios: int = Field(8000, ge=1000, le=20000)

class ReturnsResponse(BaseModel):
    daily_returns: dict
    annual_returns: dict

class SharpeResponse(BaseModel):
    sharpe_annual: dict
    stats_annual: dict

class MarkowitzResponse(BaseModel):
    frontier_points: dict
    max_sharpe: dict
    min_variance: dict
    inputs: dict

class CAPMResponse(BaseModel):
    benchmark: str
    alpha_annual: float
    beta: float
    r2: float
    regression: dict
