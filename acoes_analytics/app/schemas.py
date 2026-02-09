from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnalysisRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    benchmark: Optional[str] = None

    rf_annual: float = Field(0.10, ge=0.0, le=1.0)
    weights: Optional[List[float]] = None

    trading_days: int = Field(252, ge=200, le=270)
    n_portfolios: int = Field(8000, ge=1000, le=20000)

class ReturnsResponse(BaseModel):
    daily_returns: dict
    annual_returns: dict

class SharpeResponse(BaseModel):
    sharpe_annual: dict
    stats_annual: dict

class MarkowitzResponse(BaseModel):
    frontier_points: dict
    efficient_envelope: dict
    equal_weight: dict
    max_sharpe: dict
    min_variance: dict
    inputs: dict

class CAPMResponse(BaseModel):
    benchmark: str
    alpha_annual: float
    beta: float
    r2: float
    regression: dict
    scatter: dict

class CorrResponse(BaseModel):
    corr: dict
    cov_annual: dict

class RiskResponse(BaseModel):
    drawdown: dict
    max_drawdown: dict
    rolling_vol_21: dict
    rolling_vol_63: dict
    rolling_sharpe_63: dict
