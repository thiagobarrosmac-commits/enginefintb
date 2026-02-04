from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    benchmark: Optional[str] = None
    rf_annual: float = 0.10
    weights: Optional[List[float]] = None
    trading_days: int = 252
    n_portfolios: int = 8000

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
