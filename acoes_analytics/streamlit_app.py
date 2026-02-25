"""
QuantView — Streamlit Dashboard
=================================
Quantitative Portfolio Analytics & Risk Assessment
"""

import os
import io
import json
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date, timedelta

# =============================
# Config
# =============================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Engine TBMF— Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================
# Theme / Styling
# =============================

COLORS = ["#1B3A4B", "#065A82", "#1C7293", "#9EB3C2", "#F26419", "#E84855", "#2D936C"]
COLOR_PORT = "#F26419"  # laranja para o portfólio

quantview_template = go.layout.Template()
quantview_template.layout.colorway = COLORS
quantview_template.layout.font = dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1B3A4B")
quantview_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
quantview_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
quantview_template.layout.xaxis = dict(gridcolor="#E8ECF0", zerolinecolor="#E8ECF0")
quantview_template.layout.yaxis = dict(gridcolor="#E8ECF0", zerolinecolor="#E8ECF0")
pio.templates["quantview"] = quantview_template
pio.templates.default = "plotly_white+quantview"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1B3A4B; font-weight: 700; }
    h2, h3 { color: #065A82; }
    .stMetric label { color: #6B7280 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #1B3A4B !important; }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================

st.markdown("## 📊 Investment Engine TBMF")
st.caption("Quantitative Portfolio Analytics & Risk Assessment")

# =============================
# API Client (com retry)
# =============================

@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def call(endpoint: str, payload: dict):
    """Chama endpoint da API com retry e error handling robusto."""
    url = f"{API_URL}{endpoint}"
    try:
        r = get_session().post(url, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Não foi possível conectar à API. Verifique se o backend está online.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("⚠️ Timeout na chamada à API. O backend pode estar inicializando (cold start). Tente novamente.")
        st.stop()

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        st.error(f"Erro na API ({r.status_code}): {detail}")
        st.stop()

    return r.json()


# =============================
# Data Helpers
# =============================

def to_df_cols(d: dict, index_name: str = "ticker") -> pd.DataFrame:
    if not d:
        out = pd.DataFrame()
        out.index.name = index_name
        return out
    out = pd.DataFrame.from_dict(d, orient="columns")
    out.index.name = index_name
    return out


def series_dict_to_df(d: dict) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df.sort_index()


# =============================
# Formatadores financeiros
# =============================

def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def fmt_x(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}x"


def fmt_num(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if abs(val) >= 1e12:
        return f"${val/1e12:.1f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"{val:,.{decimals}f}"


# =============================
# Display Helpers
# =============================

def show_table(df: pd.DataFrame, title: str, fmt: str = "{:.4f}"):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info("Sem dados.")
        return
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    st.dataframe(df2.style.format(fmt), use_container_width=True)


def download_df(df: pd.DataFrame, filename_base: str):
    """Download CSV + XLSX (XLSX opcional se openpyxl indisponível)."""
    if df is None or df.empty:
        return

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label=f"📥 CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
        )

    with c2:
        try:
            import openpyxl  # noqa: F401
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="data", index=True)
            st.download_button(
                label=f"📥 Excel",
                data=output.getvalue(),
                file_name=f"{filename_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.caption("Excel indisponível (instale openpyxl)")


# =============================
# Portfolio Helpers (local)
# =============================

def normalize_weights_local(weights, n):
    if weights is None or len(weights) == 0:
        return np.ones(n) / n
    w = np.array(weights, dtype=float)
    if len(w) != n:
        raise ValueError("Pesos devem ter o mesmo tamanho dos tickers.")
    s = w.sum()
    if np.isclose(s, 0):
        raise ValueError("Soma dos pesos não pode ser zero.")
    return w / s


def portfolio_simple_returns(daily_simple: pd.DataFrame, weights) -> pd.Series:
    w = normalize_weights_local(weights, daily_simple.shape[1])
    rp = daily_simple.values @ w
    return pd.Series(rp, index=daily_simple.index, name="PORT")


def portfolio_log_returns_from_simple(port_simple: pd.Series) -> pd.Series:
    return np.log1p(port_simple).rename("PORT")


def annual_stats_from_simple(port_simple: pd.Series, trading_days=252) -> pd.Series:
    mu_d = float(port_simple.mean())
    vol_d = float(port_simple.std(ddof=1))
    ret_a = (1 + mu_d) ** trading_days - 1
    vol_a = vol_d * np.sqrt(trading_days)
    return pd.Series({"ret_annual": float(ret_a), "vol_annual": float(vol_a)})


def sharpe_from_annual(ret_annual, vol_annual, rf_annual=0.10):
    if vol_annual is None or vol_annual == 0 or np.isnan(vol_annual):
        return np.nan
    return (ret_annual - rf_annual) / vol_annual


def portfolio_drawdown_from_simple(port_simple: pd.Series) -> pd.Series:
    cum = (1 + port_simple).cumprod()
    peak = cum.cummax()
    return (cum / peak) - 1.0


def portfolio_rolling_vol(port_simple: pd.Series, window: int, trading_days=252) -> pd.Series:
    return port_simple.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def portfolio_rolling_sharpe(port_simple: pd.Series, window: int, rf_annual=0.10, trading_days=252) -> pd.Series:
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    ex = port_simple - rf_daily
    mu = ex.rolling(window).mean() * trading_days
    vol = port_simple.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    return mu / vol


# =============================
# Inputs (Sidebar-style no topo)
# =============================

with st.container():
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        tickers_raw = st.text_input("Tickers (separados por vírgula)", value="JPM, MA, V, GOOG")
    with col2:
        start = st.date_input("Início", value=date.today() - timedelta(days=365 * 5))
    with col3:
        end = st.date_input("Fim", value=date.today())

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    # Validação de datas
    if start >= end:
        st.error("Data de início deve ser anterior à data de fim.")
        st.stop()

    col4, col5, col6 = st.columns([1, 1, 2])
    with col4:
        rf_annual = st.number_input("Taxa livre de risco (anual)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    with col5:
        n_portfolios = st.number_input("Simulações Markowitz", min_value=1000, max_value=20000, value=8000, step=1000)
    with col6:
        weights_raw = st.text_input("Pesos (opcional, separados por vírgula)", value="")

weights = None
if weights_raw.strip():
    try:
        weights = [float(x.strip()) for x in weights_raw.split(",")]
    except Exception:
        st.warning("Pesos inválidos. Use números separados por vírgula (ex: 0.5, 0.3, 0.1, 0.1).")
        weights = None

# CAPM benchmark config
with st.expander("⚙️ Configuração CAPM", expanded=False):
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        benchmark_main = st.selectbox("Benchmark principal", ["Auto", "^GSPC", "^NYA", "^DJI", "^IXIC", "SPY"], index=0)
    with bcol2:
        compare_two = st.checkbox("Comparar 2 benchmarks", value=False)
    with bcol3:
        benchmark_2 = st.selectbox("Benchmark 2", ["^NYA", "^GSPC", "^DJI", "^IXIC", "SPY"], index=0)


def bench_value(sel: str):
    return None if sel == "Auto" else sel


payload_base = {
    "tickers": tickers,
    "start": str(start),
    "end": str(end),
    "rf_annual": float(rf_annual),
    "weights": weights,
    "n_portfolios": int(n_portfolios),
    "benchmark": bench_value(benchmark_main),
}

st.divider()
run = st.button("🚀 Rodar Análise", type="primary", use_container_width=True)

# =============================
# Main Analysis
# =============================

if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    # ---- Fetch base data ----
    with st.spinner("Buscando dados e calculando..."):
        ret = call("/returns", payload_base)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])

    # Portfolio series
    try:
        port_simple = portfolio_simple_returns(daily_simple, weights)
        port_log = portfolio_log_returns_from_simple(port_simple)
        w_port = normalize_weights_local(weights, len(tickers))
    except Exception as e:
        st.error(f"Erro ao calcular portfólio: {e}")
        st.stop()

    daily_simple_p = daily_simple.copy()
    daily_simple_p["PORT"] = port_simple
    daily_log_p = daily_log.copy()
    daily_log_p["PORT"] = port_log

    annual = to_df_cols(ret["annual_returns"], index_name="ticker")
    port_stats = annual_stats_from_simple(port_simple, trading_days=252)

    port_row = pd.DataFrame([{
        "ret_annual": port_stats["ret_annual"],
        "vol_annual": port_stats["vol_annual"],
        "cagr": np.nan,
    }], index=["PORT"])
    annual_p = pd.concat([annual, port_row], axis=0)
    annual_p.index.name = "ticker"

    port_sharpe = sharpe_from_annual(port_stats["ret_annual"], port_stats["vol_annual"], rf_annual=rf_annual)
    port_mdd = float(portfolio_drawdown_from_simple(port_simple).min())

    # =============================
    # KPI Cards
    # =============================

    st.markdown("### Portfolio Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Return (ann.)", fmt_pct(port_stats["ret_annual"]))
    kpi2.metric("Volatility (ann.)", fmt_pct(port_stats["vol_annual"]))
    kpi3.metric("Sharpe Ratio", fmt_x(port_sharpe))
    kpi4.metric("Max Drawdown", fmt_pct(port_mdd))
    kpi5.metric("Assets", str(len(tickers)))

    st.divider()

    # =============================
    # Tabs
    # =============================

    tab_ret, tab_risk, tab_corr, tab_mk, tab_capm, tab_var, tab_val = st.tabs([
        "📈 Returns",
        "⚡ Risk & Sharpe",
        "🔗 Correlation",
        "🎯 Markowitz",
        "📊 CAPM",
        "🛡️ VaR & Stress",
        "💰 Valuation",
    ])

    # =============================
    # TAB: Returns
    # =============================
    with tab_ret:
        c1, c2 = st.columns(2)
        with c1:
            cum = (1 + daily_simple_p).cumprod()
            fig_cum = px.line(cum, title="Cumulative Returns")
            fig_cum.update_layout(yaxis_title="Growth of $1", hovermode="x unified")
            st.plotly_chart(fig_cum, use_container_width=True)

        with c2:
            fig_log = px.line(daily_log_p, title="Daily Log Returns")
            fig_log.update_layout(yaxis_title="Log Return", hovermode="x unified")
            st.plotly_chart(fig_log, use_container_width=True)

        show_table(annual_p, "Annualized Statistics", fmt="{:.4f}")
        download_df(annual_p.round(6), "annual_stats")

        # Bar chart
        if not annual_p.empty:
            annual_plot = annual_p.copy()
            annual_plot.index.name = "ticker"
            annual_plot = annual_plot.reset_index()
            if "ticker" not in annual_plot.columns:
                annual_plot = annual_plot.rename(columns={annual_plot.columns[0]: "ticker"})
            cols_to_melt = [c for c in annual_plot.columns if c != "ticker"]
            annual_long = annual_plot.melt(id_vars="ticker", value_vars=cols_to_melt, var_name="metric", value_name="value")
            fig_bar = px.bar(annual_long, x="ticker", y="value", color="metric", barmode="group", title="Return / Vol / CAGR by Ticker")
            st.plotly_chart(fig_bar, use_container_width=True)

    # =============================
    # TAB: Risk & Sharpe
    # =============================
    with tab_risk:
        sh = call("/sharpe", payload_base)
        sharpe_df = to_df_cols(sh["sharpe_annual"], index_name="ticker")
        stats_df = to_df_cols(sh["stats_annual"], index_name="ticker")

        # Append PORT
        stats_df_p = stats_df.copy()
        stats_df_p.loc["PORT", "ret_annual"] = port_stats["ret_annual"]
        stats_df_p.loc["PORT", "vol_annual"] = port_stats["vol_annual"]

        sharpe_df_p = sharpe_df.copy()
        sharpe_df_p.loc["PORT", "sharpe_annual"] = port_sharpe

        c3, c4 = st.columns(2)
        with c3:
            show_table(sharpe_df_p, "Sharpe Ratio (annualized)", fmt="{:.4f}")
            if "sharpe_annual" in sharpe_df_p.columns and not sharpe_df_p.empty:
                fig_sh = px.bar(sharpe_df_p.reset_index(), x="ticker", y="sharpe_annual", title="Sharpe Ratio")
                fig_sh.update_traces(marker_color=[COLOR_PORT if t == "PORT" else COLORS[0] for t in sharpe_df_p.index])
                st.plotly_chart(fig_sh, use_container_width=True)

        with c4:
            show_table(stats_df_p, "Return vs Volatility", fmt="{:.4f}")
            if {"ret_annual", "vol_annual"}.issubset(stats_df_p.columns) and not stats_df_p.empty:
                fig_rv = px.scatter(
                    stats_df_p.reset_index(), x="vol_annual", y="ret_annual", text="ticker",
                    title="Risk-Return Scatter"
                )
                fig_rv.update_traces(textposition="top center", marker=dict(size=12))
                st.plotly_chart(fig_rv, use_container_width=True)

        # ---- Risk metrics (drawdown, rolling) ----
        st.divider()
        st.markdown("### Risk Metrics (including PORT)")

        rk = call("/risk", payload_base)

        dd = series_dict_to_df(rk.get("drawdown", {}))
        rv21 = series_dict_to_df(rk.get("rolling_vol_21", {}))
        rv63 = series_dict_to_df(rk.get("rolling_vol_63", {}))
        rs63 = series_dict_to_df(rk.get("rolling_sharpe_63", {}))

        dd["PORT"] = portfolio_drawdown_from_simple(port_simple)
        rv21["PORT"] = portfolio_rolling_vol(port_simple, 21, trading_days=252)
        rv63["PORT"] = portfolio_rolling_vol(port_simple, 63, trading_days=252)
        rs63["PORT"] = portfolio_rolling_sharpe(port_simple, 63, rf_annual=rf_annual, trading_days=252)

        if not dd.empty:
            fig_dd = px.line(dd, title="Drawdown")
            fig_dd.update_layout(yaxis_title="Drawdown", hovermode="x unified")
            st.plotly_chart(fig_dd, use_container_width=True)

        mdd = pd.Series(rk.get("max_drawdown", {})).sort_values()
        mdd.loc["PORT"] = port_mdd
        mdd = mdd.sort_values()
        show_table(mdd.to_frame("max_drawdown"), "Max Drawdown", fmt="{:.4f}")

        cR1, cR2 = st.columns(2)
        with cR1:
            if not rv21.empty:
                fig_v21 = px.line(rv21, title="Rolling Volatility (21d)")
                fig_v21.update_layout(hovermode="x unified")
                st.plotly_chart(fig_v21, use_container_width=True)
        with cR2:
            if not rv63.empty:
                fig_v63 = px.line(rv63, title="Rolling Volatility (63d)")
                fig_v63.update_layout(hovermode="x unified")
                st.plotly_chart(fig_v63, use_container_width=True)

        if not rs63.empty:
            fig_rs = px.line(rs63, title="Rolling Sharpe (63d)")
            fig_rs.update_layout(hovermode="x unified")
            st.plotly_chart(fig_rs, use_container_width=True)

    # =============================
    # TAB: Correlation
    # =============================
    with tab_corr:
        cc = call("/corr", payload_base)
        corr = to_df_cols(cc["corr"], index_name="ticker")
        cov_a = to_df_cols(cc["cov_annual"], index_name="ticker")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Correlation Matrix**")
            st.dataframe(corr.round(4), use_container_width=True)
            if not corr.empty:
                fig_corr = px.imshow(corr, title="Correlation Heatmap", aspect="auto",
                                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
        with cB:
            st.markdown("**Annualized Covariance**")
            st.dataframe(cov_a.round(6), use_container_width=True)
            if not cov_a.empty:
                fig_cov = px.imshow(cov_a, title="Covariance Heatmap", aspect="auto",
                                    color_continuous_scale="Viridis")
                st.plotly_chart(fig_cov, use_container_width=True)

        download_df(corr.round(6), "correlation_matrix")

    # =============================
    # TAB: Markowitz
    # =============================
    with tab_mk:
        if len(tickers) < 2:
            st.info("Markowitz requires 2+ tickers.")
        else:
            mk = call("/markowitz", payload_base)

            pts = pd.DataFrame(mk["frontier_points"])
            env = pd.DataFrame(mk.get("efficient_envelope", {}))
            eq = mk.get("equal_weight", {})
            max_sh = mk.get("max_sharpe", {})
            min_v = mk.get("min_variance", {})
            port_user_api = mk.get("port_user", {})

            # Chart
            fig_mk = px.scatter(pts, x="vol", y="ret", color="sharpe",
                                color_continuous_scale="Viridis",
                                title="Efficient Frontier (Monte Carlo)")
            fig_mk.update_traces(marker=dict(size=4, opacity=0.5))

            if not env.empty and {"vol", "ret"}.issubset(env.columns):
                fig_mk.add_trace(go.Scatter(x=env["vol"], y=env["ret"], mode="lines",
                                            name="Efficient Frontier", line=dict(color="#E84855", width=2)))

            if "vol" in eq and "ret" in eq:
                fig_mk.add_trace(go.Scatter(x=[eq["vol"]], y=[eq["ret"]], mode="markers",
                                            name="Equal Weight", marker=dict(size=14, symbol="diamond", color="#2D936C")))
            if "vol" in max_sh and "ret" in max_sh:
                fig_mk.add_trace(go.Scatter(x=[max_sh["vol"]], y=[max_sh["ret"]], mode="markers",
                                            name="Max Sharpe ⭐", marker=dict(size=14, symbol="star", color="#F26419")))
            if "vol" in min_v and "ret" in min_v:
                fig_mk.add_trace(go.Scatter(x=[min_v["vol"]], y=[min_v["ret"]], mode="markers",
                                            name="Min Variance", marker=dict(size=14, symbol="square", color="#065A82")))

            p_ret = float(port_user_api.get("ret", port_stats["ret_annual"]))
            p_vol = float(port_user_api.get("vol", port_stats["vol_annual"]))
            fig_mk.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode="markers",
                                        name="Your Portfolio", marker=dict(size=14, symbol="x", color="#E84855")))

            fig_mk.update_layout(xaxis_title="Volatility (ann.)", yaxis_title="Return (ann.)", hovermode="closest")
            st.plotly_chart(fig_mk, use_container_width=True)

            # Tables
            def weights_row_from_block(block, tickers_list):
                w = (block or {}).get("weights", {}) if isinstance(block, dict) else {}
                return {t: float(w.get(t, 0.0)) for t in tickers_list}

            markowitz_metrics = pd.DataFrame([
                {"strategy": "Equal Weight", "ret": eq.get("ret"), "vol": eq.get("vol"), "sharpe": eq.get("sharpe")},
                {"strategy": "Max Sharpe", "ret": max_sh.get("ret"), "vol": max_sh.get("vol"), "sharpe": max_sh.get("sharpe")},
                {"strategy": "Min Variance", "ret": min_v.get("ret"), "vol": min_v.get("vol"), "sharpe": min_v.get("sharpe")},
                {"strategy": "Your Portfolio", "ret": port_user_api.get("ret", p_ret), "vol": port_user_api.get("vol", p_vol),
                 "sharpe": port_user_api.get("sharpe", sharpe_from_annual(p_ret, p_vol, rf_annual=rf_annual))},
            ]).set_index("strategy")[["ret", "vol", "sharpe"]]

            markowitz_weights = pd.DataFrame({
                "Equal Weight": weights_row_from_block(eq, tickers),
                "Max Sharpe": weights_row_from_block(max_sh, tickers),
                "Min Variance": weights_row_from_block(min_v, tickers),
                "Your Portfolio": weights_row_from_block(port_user_api, tickers) if port_user_api else {t: float(w_port[i]) for i, t in enumerate(tickers)},
            }).T
            markowitz_weights.index.name = "strategy"
            markowitz_weights["total"] = markowitz_weights.sum(axis=1)

            st.markdown("**Portfolio Comparison (Return / Vol / Sharpe)**")
            st.dataframe(markowitz_metrics.round(4).style.format("{:.4f}"), use_container_width=True)
            download_df(markowitz_metrics.round(6), "markowitz_metrics")

            st.markdown("**Weight Allocation**")
            st.dataframe(markowitz_weights.round(4).style.format("{:.4f}"), use_container_width=True)
            download_df(markowitz_weights.round(6), "markowitz_weights")

    # =============================
    # TAB: CAPM
    # =============================
    with tab_capm:
        def capm_call(bench):
            p = dict(payload_base)
            p["benchmark"] = bench
            return call("/capm", p)

        capm1 = capm_call(payload_base["benchmark"])
        rows = [{
            "benchmark": capm1.get("benchmark"),
            "alpha_annual": capm1.get("alpha_annual"),
            "beta": capm1.get("beta"),
            "r2": capm1.get("r2"),
            "alpha_daily": (capm1.get("regression") or {}).get("alpha_daily"),
            "rf_daily": (capm1.get("regression") or {}).get("rf_daily"),
            "n_obs": (capm1.get("regression") or {}).get("n_obs"),
        }]

        if compare_two:
            capm2 = capm_call(bench_value(benchmark_2))
            rows.append({
                "benchmark": capm2.get("benchmark"),
                "alpha_annual": capm2.get("alpha_annual"),
                "beta": capm2.get("beta"),
                "r2": capm2.get("r2"),
                "alpha_daily": (capm2.get("regression") or {}).get("alpha_daily"),
                "rf_daily": (capm2.get("regression") or {}).get("rf_daily"),
                "n_obs": (capm2.get("regression") or {}).get("n_obs"),
            })

        capm_table = pd.DataFrame(rows).set_index("benchmark")

        st.markdown("**CAPM Regression Results**")
        st.dataframe(capm_table.round(6), use_container_width=True)
        download_df(capm_table.round(8), "capm_results")

        # Scatter
        sc = pd.DataFrame.from_dict(capm1.get("scatter", {}), orient="columns")
        if not sc.empty and {"x", "y", "y_hat"}.issubset(sc.columns):
            try:
                sc.index = pd.to_datetime(sc.index)
            except Exception:
                pass
            sc = sc.sort_index()

            fig_capm = go.Figure()
            fig_capm.add_trace(go.Scatter(x=sc["x"], y=sc["y"], mode="markers",
                                          name="Observations", marker=dict(size=4, opacity=0.5, color=COLORS[1])))
            fig_capm.add_trace(go.Scatter(x=sc["x"], y=sc["y_hat"], mode="lines",
                                          name="CAPM Line", line=dict(color="#E84855", width=2)))
            fig_capm.update_layout(
                title=f"CAPM: Portfolio vs {capm1.get('benchmark')} (excess returns)",
                xaxis_title="Benchmark Excess Return",
                yaxis_title="Portfolio Excess Return",
            )
            st.plotly_chart(fig_capm, use_container_width=True)

            # Interpretation
            beta_val = capm1.get("beta", 0)
            alpha_val = capm1.get("alpha_annual", 0)
            r2_val = capm1.get("r2", 0)
            st.markdown(f"""
            **Interpretation:** β = {beta_val:.3f} means the portfolio moves {beta_val:.1%} for every 1% move in the benchmark.
            Alpha of {fmt_pct(alpha_val)} indicates {'outperformance' if alpha_val > 0 else 'underperformance'} vs. the benchmark.
            R² = {r2_val:.1%} of portfolio variance is explained by the benchmark.
            """)

    # =============================
    # TAB: VaR & Stress (NOVO)
    # =============================
    with tab_var:
        var_data = call("/var", payload_base)

        v95 = var_data.get("var_95", {})
        v99 = var_data.get("var_99", {})
        scenarios = var_data.get("stress_scenarios", [])

        st.markdown("### Value at Risk (Portfolio)")

        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**VaR 95% Confidence**")
            var_95_df = pd.DataFrame([{
                "Metric": "VaR Daily (Parametric)",   "Value": fmt_pct(v95.get("var_daily_parametric")),
            }, {
                "Metric": "VaR Annual (Parametric)",  "Value": fmt_pct(v95.get("var_annual_parametric")),
            }, {
                "Metric": "VaR Daily (Historic)",     "Value": fmt_pct(v95.get("var_daily_historic")),
            }, {
                "Metric": "CVaR Daily (Exp. Shortfall)", "Value": fmt_pct(v95.get("cvar_daily")),
            }])
            st.table(var_95_df.set_index("Metric"))

        with vc2:
            st.markdown("**VaR 99% Confidence**")
            var_99_df = pd.DataFrame([{
                "Metric": "VaR Daily (Parametric)",   "Value": fmt_pct(v99.get("var_daily_parametric")),
            }, {
                "Metric": "VaR Annual (Parametric)",  "Value": fmt_pct(v99.get("var_annual_parametric")),
            }, {
                "Metric": "VaR Daily (Historic)",     "Value": fmt_pct(v99.get("var_daily_historic")),
            }, {
                "Metric": "CVaR Daily (Exp. Shortfall)", "Value": fmt_pct(v99.get("cvar_daily")),
            }])
            st.table(var_99_df.set_index("Metric"))

        st.markdown("### Stress Scenarios")
        if scenarios:
            stress_df = pd.DataFrame(scenarios).set_index("scenario")
            stress_df["daily_return"] = stress_df["daily_return"].apply(lambda x: fmt_pct(x, 4))
            stress_df["annual_return"] = stress_df["annual_return"].apply(lambda x: fmt_pct(x))
            st.table(stress_df)

        st.caption("VaR = maximum expected loss at a given confidence level. CVaR = average loss beyond VaR (Expected Shortfall).")

    # =============================
    # TAB: Valuation (NOVO)
    # =============================
    with tab_val:
        st.markdown("### Valuation Multiples (Peer Comparison)")

        with st.spinner("Fetching valuation data..."):
            val_data = call("/valuation", payload_base)

        multiples = to_df_cols(val_data.get("multiples", {}), index_name="ticker")

        if not multiples.empty:
            # Key multiples table
            key_cols = ["ev_ebitda", "pe_trailing", "pe_forward", "ev_revenue", "pb", "ps"]
            available_key = [c for c in key_cols if c in multiples.columns]

            if available_key:
                st.markdown("**Key Multiples**")
                key_df = multiples[available_key].copy()
                key_df.columns = [c.upper().replace("_", " ") for c in key_df.columns]

                # Add sector mean row
                numeric_key = key_df.apply(pd.to_numeric, errors="coerce")
                mean_row = numeric_key.mean().to_frame("PEER MEAN").T
                key_display = pd.concat([numeric_key, mean_row])

                st.dataframe(key_display.round(2).style.format("{:.2f}", na_rep="—"), use_container_width=True)

                # Chart — EV/EBITDA comparison
                if "ev_ebitda" in multiples.columns:
                    ev_data = multiples["ev_ebitda"].dropna()
                    if not ev_data.empty:
                        fig_ev = px.bar(
                            ev_data.reset_index(), x="ticker", y="ev_ebitda",
                            title="EV/EBITDA Comparison",
                            labels={"ev_ebitda": "EV/EBITDA", "ticker": ""},
                        )
                        peer_mean = float(ev_data.mean())
                        fig_ev.add_hline(y=peer_mean, line_dash="dash", line_color="#E84855",
                                         annotation_text=f"Peer Mean: {peer_mean:.1f}x")
                        st.plotly_chart(fig_ev, use_container_width=True)

            # Fundamentals table
            fund_cols = ["profit_margin", "revenue_growth", "gross_margin", "operating_margin", "roe", "roa", "debt_to_equity", "current_ratio"]
            available_fund = [c for c in fund_cols if c in multiples.columns]

            if available_fund:
                st.markdown("**Fundamentals**")
                fund_df = multiples[available_fund].copy()
                fund_df.columns = [c.replace("_", " ").title() for c in fund_df.columns]
                st.dataframe(fund_df.round(4).style.format("{:.4f}", na_rep="—"), use_container_width=True)

            # Sector / Industry
            meta_cols = ["sector", "industry", "market_cap", "enterprise_value"]
            available_meta = [c for c in meta_cols if c in multiples.columns]
            if available_meta:
                st.markdown("**Company Overview**")
                meta_df = multiples[available_meta].copy()
                for mc in ["market_cap", "enterprise_value"]:
                    if mc in meta_df.columns:
                        meta_df[mc] = meta_df[mc].apply(lambda x: fmt_num(x) if pd.notna(x) else "—")
                st.dataframe(meta_df, use_container_width=True)

            download_df(multiples.round(4), "valuation_multiples")
        else:
            st.info("No valuation data available for these tickers.")
