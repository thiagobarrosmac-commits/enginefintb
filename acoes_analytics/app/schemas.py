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
    daily_returns: Dict[str, Any]
    annual_returns: Dict[str, Any]


class SharpeResponse(BaseModel):
    sharpe_annual: Dict[str, Any]
    stats_annual: Dict[str, Any]


class CorrResponse(BaseModel):
    corr: Dict[str, Any]
    cov_annual: Dict[str, Any]


class RiskResponse(BaseModel):
    drawdown: Dict[str, Any]
    max_drawdown: Dict[str, Any]
    rolling_vol_21: Dict[str, Any]
    rolling_vol_63: Dict[str, Any]
    rolling_sharpe_63: Dict[str, Any]


class MarkowitzResponse(BaseModel):
    frontier_points: Dict[str, Any]
    efficient_envelope: Dict[str, Any]
    equal_weight: Dict[str, Any]
    max_sharpe: Dict[str, Any]
    min_variance: Dict[str, Any]

    # ✅ NOVO: estatísticas do PORT do usuário calculadas no backend
    port_user: Dict[str, Any]

    inputs: Dict[str, Any]


class CAPMResponse(BaseModel):
    benchmark: str
    alpha_annual: float
    beta: float
    r2: float
    regression: Dict[str, Any]
    scatter: Dict[str, Any]
