import os
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

def call(endpoint: str, payload: dict):
    url = f"{API_URL}{endpoint}"
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        st.error(f"Erro na API ({r.status_code}) em {url}")
        st.code(r.text)
        raise RuntimeError(r.text)
    return r.json()

def to_df_cols(d: dict, index_name="ticker") -> pd.DataFrame:
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

def show_table(df: pd.DataFrame, title: str, fmt: str = "{:.4f}"):
    st.markdown(f"**{title}**")
    if df is None or df.empty:
        st.info("Sem dados.")
        return
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    st.table(df2.style.format(fmt))

# Inputs
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    tickers_raw = st.text_input("Tickers (separados por vírgula)", value="JPM,MA,V")
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
    weights_raw = st.text_input("Pesos (opcional)", value="")

weights = None
if weights_raw.strip():
    try:
        weights = [float(x.strip()) for x in weights_raw.split(",")]
    except Exception:
        st.warning("Pesos inválidos. Use números separados por vírgula (ex: 0.5,0.5).")

payload = {
    "tickers": tickers,
    "start": str(start),
    "end": str(end),
    "rf_annual": float(rf_annual),
    "weights": weights,
    "n_portfolios": int(n_portfolios),
    "benchmark": None
}

run = st.button("Rodar análise")

if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    # Returns
    st.subheader("1) Retorno diário + 2) Retorno anual")
    ret = call("/returns", payload)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])
    annual = to_df_cols(ret["annual_returns"], index_name="ticker")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset: Retorno diário (simples)**")
        st.dataframe(daily_simple.tail(20), use_container_width=True, height=240)
        cum = (1 + daily_simple).cumprod()
        st.plotly_chart(px.line(cum, title="Retorno acumulado (simples)"), use_container_width=True)

    with c2:
        st.markdown("**Dataset: Retorno diário (log)**")
        st.dataframe(daily_log.tail(20), use_container_width=True, height=240)
        st.plotly_chart(px.line(daily_log, title="Retorno diário (log)"), use_container_width=True)

    show_table(annual, "Estatísticas anualizadas (ret_annual, vol_annual, cagr se disponível)")

    if not annual.empty:
        annual_long = annual.reset_index().melt(id_vars="ticker", var_name="metric", value_name="value")
        st.plotly_chart(
            px.bar(annual_long, x="ticker", y="value", color="metric", barmode="group",
                   title="Retorno/Vol/CAGR anual (por ticker)"),
            use_container_width=True
        )

    # Sharpe
    st.divider()
    st.subheader("3) Sharpe")
    sh = call("/sharpe", payload)

    sharpe_df = to_df_cols(sh["sharpe_annual"], index_name="ticker")
    stats_df = to_df_cols(sh["stats_annual"], index_name="ticker")

    c3, c4 = st.columns(2)
    with c3:
        show_table(sharpe_df, "Sharpe anual")
        if "sharpe_annual" in sharpe_df.columns and not sharpe_df.empty:
            st.plotly_chart(px.bar(sharpe_df.reset_index(), x="ticker", y="sharpe_annual", title="Sharpe anual"),
                            use_container_width=True)

    with c4:
        show_table(stats_df, "Stats anual (ret/vol)")
        if {"ret_annual", "vol_annual"}.issubset(stats_df.columns) and not stats_df.empty:
            st.plotly_chart(
                px.scatter(stats_df.reset_index(), x="vol_annual", y="ret_annual", text="ticker",
                           title="Risco x Retorno (anual)"),
                use_container_width=True
            )

    # Corr / Cov
    st.divider()
    st.subheader("Matriz de Correlação e Covariância Anualizada")
    cc = call("/corr", payload)
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

    # Risk
    st.divider()
    st.subheader("Risco: Drawdown e Métricas Rolling")
    rk = call("/risk", payload)

    dd = series_dict_to_df(rk.get("drawdown", {}))
    st.markdown("**Drawdown**")
    if not dd.empty:
        st.plotly_chart(px.line(dd, title="Drawdown por Ativo"), use_container_width=True)

    mdd = pd.Series(rk.get("max_drawdown", {})).sort_values()
    st.markdown("**Max Drawdown**")
    if not mdd.empty:
        st.table(mdd.to_frame("max_drawdown").style.format("{:.4f}"))

    rv21 = series_dict_to_df(rk.get("rolling_vol_21", {}))
    rv63 = series_dict_to_df(rk.get("rolling_vol_63", {}))
    rs63 = series_dict_to_df(rk.get("rolling_sharpe_63", {}))

    cR1, cR2 = st.columns(2)
    with cR1:
        st.markdown("**Vol rolling 21d**")
        if not rv21.empty:
            st.plotly_chart(px.line(rv21, title="Vol Rolling 21d (anualizada)"), use_container_width=True)
    with cR2:
        st.markdown("**Vol rolling 63d**")
        if not rv63.empty:
            st.plotly_chart(px.line(rv63, title="Vol Rolling 63d (anualizada)"), use_container_width=True)

    st.markdown("**Sharpe rolling 63d**")
    if not rs63.empty:
        st.plotly_chart(px.line(rs63, title="Sharpe Rolling 63d"), use_container_width=True)

    # Markowitz
    st.divider()
    st.subheader("4) Markowitz")
    if len(tickers) < 2:
        st.info("Markowitz requer 2+ tickers. Pulei este módulo.")
    else:
        mk = call("/markowitz", payload)

        pts = pd.DataFrame(mk["frontier_points"])
        env = pd.DataFrame(mk.get("efficient_envelope", {}))
        eq = mk.get("equal_weight", {})
        max_sh = mk.get("max_sharpe", {})
        min_v = mk.get("min_variance", {})

        fig = px.scatter(pts, x="vol", y="ret", hover_data=["sharpe"], title="Markowitz: Simulação + Envelope")
        if not env.empty and {"vol", "ret"}.issubset(env.columns):
            fig.add_trace(go.Scatter(x=env["vol"], y=env["ret"], mode="lines", name="Envelope"))
        if "vol" in eq and "ret" in eq:
            fig.add_trace(go.Scatter(x=[eq["vol"]], y=[eq["ret"]], mode="markers", name="Equal Weight"))
        if "vol" in max_sh and "ret" in max_sh:
            fig.add_trace(go.Scatter(x=[max_sh["vol"]], y=[max_sh["ret"]], mode="markers", name="Max Sharpe"))

        st.plotly_chart(fig, use_container_width=True)
        st.json({"equal_weight": eq, "max_sharpe": max_sh, "min_variance": min_v})

    # CAPM
    st.divider()
    st.subheader("5) CAPM do Portfólio (alpha, beta, R²)")
    capm = call("/capm", payload)

    st.json({
        "benchmark": capm.get("benchmark"),
        "alpha_annual": capm.get("alpha_annual"),
        "beta": capm.get("beta"),
        "r2": capm.get("r2"),
        "regression": capm.get("regression")
    })

    sc = pd.DataFrame.from_dict(capm.get("scatter", {}), orient="columns")
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
                title="CAPM: Benchmark excess vs Portfolio excess",
                xaxis_title="Benchmark excess",
                yaxis_title="Portfolio excess"
            )
            st.plotly_chart(fig, use_container_width=True)
