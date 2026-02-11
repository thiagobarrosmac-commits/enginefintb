import os
import io
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import date, timedelta

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Análise de Ações", layout="wide")
st.title("Análise Quantitativa de Ações (FastAPI + Streamlit)")
st.caption(f"API_URL em uso: {API_URL}")

# -----------------------------
# API + Data helpers
# -----------------------------
def call(endpoint: str, payload: dict):
    url = f"{API_URL}{endpoint}"
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        st.error(f"Erro na API ({r.status_code}) em {url}")
        st.code(r.text)
        raise RuntimeError(r.text)
    return r.json()

def to_df_cols(d: dict, index_name="ticker") -> pd.DataFrame:
    # backend usa pandas .to_dict() default => {col: {idx: val}}
    if not d:
        out = pd.DataFrame()
        out.index.name = index_name
        return out
    out = pd.DataFrame.from_dict(d, orient="columns")
    out.index.name = index_name
    return out

def series_dict_to_df(d: dict) -> pd.DataFrame:
    # dict {col: {timestamp: value}}
    if not d:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df.sort_index()

def show_table(df: pd.DataFrame, title: str, fmt: str = "{:.6f}"):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info("Sem dados.")
        return
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    st.table(df2.style.format(fmt))

def download_df(df: pd.DataFrame, filename_base: str):
    """Botões CSV + XLSX para Excel. XLSX só se openpyxl estiver disponível."""
    if df is None or df.empty:
        return

    # CSV (sempre)
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=f"⬇️ Baixar {filename_base}.csv",
        data=csv_bytes,
        file_name=f"{filename_base}.csv",
        mime="text/csv",
    )

    # XLSX (opcional)
    try:
        import openpyxl  # noqa: F401

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data", index=True)

        st.download_button(
            label=f"⬇️ Baixar {filename_base}.xlsx",
            data=output.getvalue(),
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("ℹ️ XLSX indisponível (openpyxl não instalado). Use CSV ou adicione openpyxl ao requirements.txt.")

# -----------------------------
# Portfolio helpers (local)
# -----------------------------
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
    return (mu / vol)

# -----------------------------
# Inputs
# -----------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    tickers_raw = st.text_input("Tickers (separados por vírgula)", value="JPM,MA,V,GOOG")
with col2:
    start = st.date_input("Início", value=date.today() - timedelta(days=365 * 5))
with col3:
    end = st.date_input("Fim", value=date.today())

tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

col4, col5, col6 = st.columns([1, 1, 2])
with col4:
    rf_annual = st.number_input("RF anual", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
with col5:
    n_portfolios = st.number_input("N carteiras (Markowitz)", min_value=1000, max_value=20000, value=8000, step=1000)
with col6:
    weights_raw = st.text_input("Pesos (opcional, separados por vírgula)", value="")

weights = None
if weights_raw.strip():
    try:
        weights = [float(x.strip()) for x in weights_raw.split(",")]
    except Exception:
        st.warning("Pesos inválidos. Use números separados por vírgula (ex: 0.5,0.5).")
        weights = None

st.divider()
st.subheader("CAPM: Benchmark")
bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 1.6])

with bcol1:
    benchmark_main = st.selectbox(
        "Benchmark principal (CAPM)",
        ["Auto", "^GSPC", "^NYA", "^DJI", "^IXIC", "SPY"],
        index=0
    )

with bcol2:
    compare_two = st.checkbox("Comparar com 2 benchmarks", value=True)

with bcol3:
    benchmark_2 = st.selectbox(
        "Benchmark 2 (se comparar)",
        ["^NYA", "^GSPC", "^DJI", "^IXIC", "SPY"],
        index=0
    )

run = st.button("Rodar análise")

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

# -----------------------------
# Run
# -----------------------------
if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    # -----------------------------
    # 1) Returns
    # -----------------------------
    st.subheader("1) Retorno diário + 2) Retorno anual")
    ret = call("/returns", payload_base)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])

    # Portfolio series (PORT)
    try:
        port_simple = portfolio_simple_returns(daily_simple, weights)
        port_log = portfolio_log_returns_from_simple(port_simple)
        w_port = normalize_weights_local(weights, len(tickers))
    except Exception as e:
        st.error("Erro ao calcular PORTFÓLIO com os pesos informados.")
        st.exception(e)
        st.stop()

    daily_simple_p = daily_simple.copy()
    daily_simple_p["PORT"] = port_simple

    daily_log_p = daily_log.copy()
    daily_log_p["PORT"] = port_log

    annual = to_df_cols(ret["annual_returns"], index_name="ticker")

    # PORT anual (para tabelas/pontos)
    port_stats = annual_stats_from_simple(port_simple, trading_days=252)
    port_row = pd.DataFrame([{
        "ret_annual": port_stats["ret_annual"],
        "vol_annual": port_stats["vol_annual"],
        "cagr": np.nan
    }], index=["PORT"])

    annual_p = pd.concat([annual, port_row], axis=0)
    annual_p.index.name = "ticker"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset: Retorno diário (simples)**")
        st.dataframe(daily_simple_p.tail(20), use_container_width=True, height=240)
        cum = (1 + daily_simple_p).cumprod()
        st.plotly_chart(px.line(cum, title="Retorno acumulado (simples) + PORT"), use_container_width=True)

    with c2:
        st.markdown("**Dataset: Retorno diário (log)**")
        st.dataframe(daily_log_p.tail(20), use_container_width=True, height=240)
        st.plotly_chart(px.line(daily_log_p, title="Retorno diário (log) + PORT"), use_container_width=True)

    show_table(annual_p, "Estatísticas anualizadas (tickers + PORT)", fmt="{:.6f}")

    # Gráfico anual (melt robusto)
    if not annual_p.empty:
        annual_plot = annual_p.copy()
        annual_plot.index.name = "ticker"
        annual_plot = annual_plot.reset_index()
        if "ticker" not in annual_plot.columns:
            annual_plot = annual_plot.rename(columns={annual_plot.columns[0]: "ticker"})
        annual_long = annual_plot.melt(id_vars="ticker", var_name="metric", value_name="value")

        st.plotly_chart(
            px.bar(
                annual_long,
                x="ticker",
                y="value",
                color="metric",
                barmode="group",
                title="Retorno/Vol/CAGR anual (por ticker) + PORT"
            ),
            use_container_width=True
        )

    # -----------------------------
    # 3) Sharpe
    # -----------------------------
    st.divider()
    st.subheader("3) Sharpe")
    sh = call("/sharpe", payload_base)

    sharpe_df = to_df_cols(sh["sharpe_annual"], index_name="ticker")
    stats_df = to_df_cols(sh["stats_annual"], index_name="ticker")

    # Append PORT
    stats_df_p = stats_df.copy()
    stats_df_p.loc["PORT", "ret_annual"] = port_stats["ret_annual"]
    stats_df_p.loc["PORT", "vol_annual"] = port_stats["vol_annual"]

    port_sh = sharpe_from_annual(port_stats["ret_annual"], port_stats["vol_annual"], rf_annual=rf_annual)
    sharpe_df_p = sharpe_df.copy()
    sharpe_df_p.loc["PORT", "sharpe_annual"] = port_sh

    c3, c4 = st.columns(2)
    with c3:
        show_table(sharpe_df_p, "Sharpe anual (tickers + PORT)", fmt="{:.6f}")
        if "sharpe_annual" in sharpe_df_p.columns and not sharpe_df_p.empty:
            st.plotly_chart(
                px.bar(sharpe_df_p.reset_index(), x="ticker", y="sharpe_annual", title="Sharpe anual (inclui PORT)"),
                use_container_width=True
            )

    with c4:
        show_table(stats_df_p, "Stats anual (ret/vol) (tickers + PORT)", fmt="{:.6f}")
        if {"ret_annual", "vol_annual"}.issubset(stats_df_p.columns) and not stats_df_p.empty:
            st.plotly_chart(
                px.scatter(
                    stats_df_p.reset_index(),
                    x="vol_annual",
                    y="ret_annual",
                    text="ticker",
                    title="Risco x Retorno (anual) (inclui PORT)"
                ),
                use_container_width=True
            )

    # -----------------------------
    # Corr / Cov
    # -----------------------------
    st.divider()
    st.subheader("Matriz de Correlação e Covariância Anualizada")
    cc = call("/corr", payload_base)
    corr = to_df_cols(cc["corr"], index_name="ticker")
    cov_a = to_df_cols(cc["cov_annual"], index_name="ticker")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Correlação**")
        st.dataframe(corr, use_container_width=True, height=260)
        if not corr.empty:
            st.plotly_chart(px.imshow(corr, title="Heatmap de Correlação", aspect="auto"), use_container_width=True)
    with cB:
        st.markdown("**Covariância anualizada**")
        st.dataframe(cov_a, use_container_width=True, height=260)
        if not cov_a.empty:
            st.plotly_chart(px.imshow(cov_a, title="Heatmap de Covariância Anualizada", aspect="auto"),
                            use_container_width=True)

    # -----------------------------
    # Risk (API) + PORT (local)
    # -----------------------------
    st.divider()
    st.subheader("Risco: Drawdown e Métricas Rolling (inclui PORT)")
    rk = call("/risk", payload_base)

    dd = series_dict_to_df(rk.get("drawdown", {}))
    rv21 = series_dict_to_df(rk.get("rolling_vol_21", {}))
    rv63 = series_dict_to_df(rk.get("rolling_vol_63", {}))
    rs63 = series_dict_to_df(rk.get("rolling_sharpe_63", {}))

    dd["PORT"] = portfolio_drawdown_from_simple(port_simple)
    rv21["PORT"] = portfolio_rolling_vol(port_simple, 21, trading_days=252)
    rv63["PORT"] = portfolio_rolling_vol(port_simple, 63, trading_days=252)
    rs63["PORT"] = portfolio_rolling_sharpe(port_simple, 63, rf_annual=rf_annual, trading_days=252)

    st.markdown("**Drawdown**")
    if not dd.empty:
        st.plotly_chart(px.line(dd, title="Drawdown por Ativo (inclui PORT)"), use_container_width=True)

    mdd = pd.Series(rk.get("max_drawdown", {})).sort_values()
    mdd.loc["PORT"] = float(dd["PORT"].min()) if "PORT" in dd.columns else np.nan
    mdd = mdd.sort_values()

    st.markdown("**Max Drawdown**")
    if not mdd.empty:
        st.table(mdd.to_frame("max_drawdown").style.format("{:.6f}"))

    cR1, cR2 = st.columns(2)
    with cR1:
        st.markdown("**Vol rolling 21d (anualizada)**")
        if not rv21.empty:
            st.plotly_chart(px.line(rv21, title="Vol Rolling 21d (inclui PORT)"), use_container_width=True)
    with cR2:
        st.markdown("**Vol rolling 63d (anualizada)**")
        if not rv63.empty:
            st.plotly_chart(px.line(rv63, title="Vol Rolling 63d (inclui PORT)"), use_container_width=True)

    st.markdown("**Sharpe rolling 63d**")
    if not rs63.empty:
        st.plotly_chart(px.line(rs63, title="Sharpe Rolling 63d (inclui PORT)"), use_container_width=True)

    # -----------------------------
    # 4) Markowitz (inclui ponto PORT)
    # -----------------------------
    st.divider()
    st.subheader("4) Markowitz (inclui ponto PORT)")

    if len(tickers) < 2:
        st.info("Markowitz requer 2+ tickers. Pulei este módulo.")
    else:
        mk = call("/markowitz", payload_base)

        pts = pd.DataFrame(mk["frontier_points"])
        env = pd.DataFrame(mk.get("efficient_envelope", {}))
        eq = mk.get("equal_weight", {})
        max_sh = mk.get("max_sharpe", {})
        min_v = mk.get("min_variance", {})
        port_user_api = mk.get("port_user", {})  # ✅ novo (backend patch)

        fig = px.scatter(pts, x="vol", y="ret", hover_data=["sharpe"], title="Markowitz: Simulação + Envelope + PORT")
        if not env.empty and {"vol", "ret"}.issubset(env.columns):
            fig.add_trace(go.Scatter(x=env["vol"], y=env["ret"], mode="lines", name="Envelope"))
        if "vol" in eq and "ret" in eq:
            fig.add_trace(go.Scatter(x=[eq["vol"]], y=[eq["ret"]], mode="markers", name="Equal Weight"))
        if "vol" in max_sh and "ret" in max_sh:
            fig.add_trace(go.Scatter(x=[max_sh["vol"]], y=[max_sh["ret"]], mode="markers", name="Max Sharpe"))

        # PORT point — prioriza backend (port_user), fallback para local
        p_ret = float(port_user_api.get("ret", port_stats["ret_annual"]))
        p_vol = float(port_user_api.get("vol", port_stats["vol_annual"]))
        fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode="markers", name="PORT (seus pesos)"))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # MARKOWITZ -> TABELAS (Excel-friendly)
        # -----------------------------
        def weights_row_from_block(block, tickers_list):
            w = (block or {}).get("weights", {}) if isinstance(block, dict) else {}
            return {t: float(w.get(t, 0.0)) for t in tickers_list}

        # ✅ usa min_variance ret/sharpe do backend se existir
        markowitz_metrics = pd.DataFrame(
            [
                {"strategy": "equal_weight", "ret": eq.get("ret"), "vol": eq.get("vol"), "sharpe": eq.get("sharpe")},
                {"strategy": "max_sharpe", "ret": max_sh.get("ret"), "vol": max_sh.get("vol"), "sharpe": max_sh.get("sharpe")},
                {"strategy": "min_variance", "ret": min_v.get("ret"), "vol": min_v.get("vol"), "sharpe": min_v.get("sharpe")},
                {"strategy": "port_user", "ret": port_user_api.get("ret", p_ret), "vol": port_user_api.get("vol", p_vol), "sharpe": port_user_api.get("sharpe", sharpe_from_annual(p_ret, p_vol, rf_annual=rf_annual))},
            ]
        ).set_index("strategy")

        # ordena colunas para evitar confusão no print
        markowitz_metrics = markowitz_metrics[["ret", "vol", "sharpe"]]

        # pesos: usa port_user do backend, fallback para pesos locais
        markowitz_weights = pd.DataFrame(
            {
                "equal_weight": weights_row_from_block(eq, tickers),
                "max_sharpe": weights_row_from_block(max_sh, tickers),
                "min_variance": weights_row_from_block(min_v, tickers),
                "port_user": weights_row_from_block(port_user_api, tickers) if port_user_api else {t: float(w_port[i]) for i, t in enumerate(tickers)},
            }
        ).T
        markowitz_weights.index.name = "strategy"
        markowitz_weights["sum_weights"] = markowitz_weights.sum(axis=1)

        st.markdown("### Markowitz — Resumo (ret/vol/sharpe)")
        st.dataframe(markowitz_metrics.round(6), use_container_width=True)
        download_df(markowitz_metrics.round(6), "markowitz_metrics")

        st.markdown("### Markowitz — Pesos (por estratégia)")
        st.dataframe(markowitz_weights.round(6), use_container_width=True)
        download_df(markowitz_weights.round(6), "markowitz_weights")

        # dica de debugging
        if "port_user" not in mk:
            st.warning("Seu backend ainda não está retornando 'port_user'. Atualize o serviço do Render com o patch do backend.")

    # -----------------------------
    # 5) CAPM do Portfólio (alpha, beta, R²) -> TABELAS
    # -----------------------------
    st.divider()
    st.subheader("5) CAPM do Portfólio (alpha, beta, R²)")

    def capm_call(bench):
        p = dict(payload_base)
        p["benchmark"] = bench
        return call("/capm", p)

    capm1 = capm_call(payload_base["benchmark"])  # pode ser None (Auto)
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

    st.markdown("### CAPM — Tabela (Excel-friendly)")
    st.dataframe(capm_table.round(8), use_container_width=True)
    download_df(capm_table.round(8), "capm_table")

    # Scatter CAPM (benchmark principal)
    st.markdown("### CAPM — Scatter (benchmark principal)")
    sc = pd.DataFrame.from_dict(capm1.get("scatter", {}), orient="columns")
    if not sc.empty:
        try:
            sc.index = pd.to_datetime(sc.index)
        except Exception:
            pass
        sc = sc.sort_index()

        if {"x", "y", "y_hat"}.issubset(sc.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["y"], mode="markers", name="Obs"))
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["y_hat"], mode="lines", name="Linha CAPM"))
            fig.update_layout(
                title=f"CAPM: Benchmark excess vs Portfolio excess ({capm1.get('benchmark')})",
                xaxis_title="Benchmark excess",
                yaxis_title="Portfolio excess"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scatter CAPM não contém colunas esperadas (x, y, y_hat).")
    else:
        st.info("Sem dados de scatter CAPM.")
