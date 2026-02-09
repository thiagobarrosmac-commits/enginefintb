import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import date, timedelta

# =============================
# Config / Env
# =============================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Análise de Ações", layout="wide")
st.title("Análise Quantitativa de Ações (FastAPI + Streamlit)")
st.caption(f"API_URL em uso: {API_URL}")

# =============================
# Helpers
# =============================
def call(endpoint: str, payload: dict):
    url = f"{API_URL}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        st.error(f"Falha ao conectar na API: {url}")
        st.exception(e)
        raise

    if r.status_code != 200:
        st.error(f"Erro na API ({r.status_code}) em {url}")
        st.code(r.text)
        raise RuntimeError(r.text)

    return r.json()

def to_df_indexed(d: dict, index_name="ticker") -> pd.DataFrame:
    """Converte dict->DataFrame assumindo tickers no index (orient=index)."""
    if d is None:
        out = pd.DataFrame()
    else:
        out = pd.DataFrame.from_dict(d, orient="index")
    out.index.name = index_name
    return out

def series_dict_to_df(d: dict) -> pd.DataFrame:
    """Converte dict {col: {timestamp: value}} em DataFrame com index datetime."""
    if not d:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df = df.sort_index()
    return df

# =============================
# Inputs
# =============================
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
    rf_annual = st.number_input(
        "RF anual (ex: 0.10)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )
with col5:
    n_portfolios = st.number_input(
        "N carteiras (Markowitz)",
        min_value=1000,
        max_value=20000,
        value=8000,
        step=1000
    )
with col6:
    weights_raw = st.text_input("Pesos (opcional, separados por vírgula)", value="")

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

# =============================
# Run
# =============================
if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    # -----------------------------
    # 1) Retorno diário + 2) anual
    # -----------------------------
    st.subheader("1) Retorno diário + 2) Retorno anual")
    ret = call("/returns", payload)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])

    # ✅ Correção definitiva: annual_returns orient=index
    annual = to_df_indexed(ret["annual_returns"], index_name="ticker")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset: Retorno diário (simples)**")
        st.dataframe(daily_simple.tail(20), use_container_width=True)

        cum = (1 + daily_simple).cumprod()
        fig = px.line(cum, title="Retorno acumulado (simples)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Dataset: Retorno diário (log)**")
        st.dataframe(daily_log.tail(20), use_container_width=True)

        fig = px.line(daily_log, title="Retorno diário (log)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Dataset: Estatísticas anualizadas (retorno e volatilidade)**")
    st.dataframe(annual, use_container_width=True)

    if not annual.empty:
        annual_long = annual.reset_index().melt(id_vars="ticker", var_name="metric", value_name="value")
        fig = px.bar(
            annual_long,
            x="ticker",
            y="value",
            color="metric",
            barmode="group",
            title="Retorno e Volatilidade anual (por ticker)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 3) Sharpe
    # -----------------------------
    st.divider()
    st.subheader("3) Sharpe")

    sh = call("/sharpe", payload)

    # ✅ Correção definitiva: sharpe e stats orient=index
    sharpe_df = to_df_indexed(sh["sharpe_annual"], index_name="ticker")
    stats_df = to_df_indexed(sh["stats_annual"], index_name="ticker")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Dataset: Sharpe anual**")
        st.dataframe(sharpe_df, use_container_width=True)

        if "sharpe_annual" in sharpe_df.columns and not sharpe_df.empty:
            fig = px.bar(
                sharpe_df.reset_index(),
                x="ticker",
                y="sharpe_annual",
                title="Sharpe anual"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sharpe vazio/sem coluna esperada.")

    with c4:
        st.markdown("**Dataset: Stats anual (ret/vol)**")
        st.dataframe(stats_df, use_container_width=True)

        if {"ret_annual", "vol_annual"}.issubset(stats_df.columns) and not stats_df.empty:
            fig = px.scatter(
                stats_df.reset_index(),
                x="vol_annual",
                y="ret_annual",
                title="Risco x Retorno (anual)",
                text="ticker"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stats anual sem colunas esperadas (ret_annual / vol_annual).")

    # -----------------------------
    # Correlação + Covariância
    # -----------------------------
    st.divider()
    st.subheader("Matriz de Correlação e Covariância Anualizada")

    cc = call("/corr", payload)
    corr = to_df_indexed(cc["corr"], index_name="ticker")
    cov_a = to_df_indexed(cc["cov_annual"], index_name="ticker")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Correlação**")
        st.dataframe(corr, use_container_width=True)
        if not corr.empty:
            fig = px.imshow(corr, title="Heatmap de Correlação", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.markdown("**Covariância anualizada**")
        st.dataframe(cov_a, use_container_width=True)
        if not cov_a.empty:
            fig = px.imshow(cov_a, title="Heatmap de Covariância Anualizada", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Risco: drawdown + rolling
    # -----------------------------
    st.divider()
    st.subheader("Risco: Drawdown e Métricas Rolling")

    rk = call("/risk", payload)

    dd = series_dict_to_df(rk.get("drawdown", {}))
    st.markdown("**Drawdown**")
    if not dd.empty:
        st.dataframe(dd.tail(20), use_container_width=True)
        fig = px.line(dd, title="Drawdown por Ativo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados de drawdown.")

    mdd = pd.Series(rk.get("max_drawdown", {})).sort_values()
    st.markdown("**Max Drawdown (pior queda)**")
    if not mdd.empty:
        st.dataframe(mdd.to_frame("max_drawdown"), use_container_width=True)
        fig = px.bar(
            mdd.reset_index().rename(columns={"index": "ticker", 0: "max_drawdown"}),
            x="ticker",
            y="max_drawdown",
            title="Max Drawdown por Ativo"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados de max drawdown.")

    rv21 = series_dict_to_df(rk.get("rolling_vol_21", {}))
    rv63 = series_dict_to_df(rk.get("rolling_vol_63", {}))
    rs63 = series_dict_to_df(rk.get("rolling_sharpe_63", {}))

    cR1, cR2 = st.columns(2)
    with cR1:
        st.markdown("**Volatilidade rolling (21d) anualizada**")
        if not rv21.empty:
            fig = px.line(rv21, title="Vol Rolling 21d")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados de vol rolling 21d.")

    with cR2:
        st.markdown("**Volatilidade rolling (63d) anualizada**")
        if not rv63.empty:
            fig = px.line(rv63, title="Vol Rolling 63d")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados de vol rolling 63d.")

    st.markdown("**Sharpe rolling (63d)**")
    if not rs63.empty:
        fig = px.line(rs63, title="Sharpe Rolling 63d")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados de sharpe rolling 63d.")

    # -----------------------------
    # 4) Markowitz (com envelope + equal-weight)
    # -----------------------------
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

        c5, c6 = st.columns([2, 1])
        with c5:
            fig = px.scatter(
                pts,
                x="vol",
                y="ret",
                title="Markowitz: Simulação + Envelope (fronteira aprox.)",
                hover_data=["sharpe"]
            )

            # envelope (linha)
            if not env.empty and {"vol", "ret"}.issubset(env.columns):
                fig.add_trace(go.Scatter(
                    x=env["vol"], y=env["ret"],
                    mode="lines",
                    name="Envelope (Fronteira aprox.)"
                ))

            # equal-weight
            if "vol" in eq and "ret" in eq:
                fig.add_trace(go.Scatter(
                    x=[eq["vol"]], y=[eq["ret"]],
                    mode="markers",
                    name="Equal Weight"
                ))

            # max sharpe
            if "vol" in max_sh and "ret" in max_sh:
                fig.add_trace(go.Scatter(
                    x=[max_sh["vol"]], y=[max_sh["ret"]],
                    mode="markers",
                    name="Max Sharpe"
                ))

            st.plotly_chart(fig, use_container_width=True)

        with c6:
            st.markdown("**Equal Weight**")
            st.json(eq)
            st.markdown("**Max Sharpe**")
            st.json(max_sh)
            st.markdown("**Min Variância (otimização)**")
            st.json(min_v)

    # -----------------------------
    # 5) CAPM (scatter + linha)
    # -----------------------------
    st.divider()
    st.subheader("5) CAPM do Portfólio (alpha, beta, R²)")

    capm = call("/capm", payload)

    # mostra resumo
    st.json({
        "benchmark": capm.get("benchmark"),
        "alpha_annual": capm.get("alpha_annual"),
        "beta": capm.get("beta"),
        "r2": capm.get("r2"),
        "regression": capm.get("regression")
    })

    sc = pd.DataFrame.from_dict(capm.get("scatter", {}), orient="index")
    if not sc.empty:
        try:
            sc.index = pd.to_datetime(sc.index)
        except Exception:
            pass
        sc = sc.sort_index()

        # scatter x vs y e linha ajustada
        fig = go.Figure()
        if {"x", "y", "y_hat"}.issubset(sc.columns):
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["y"], mode="markers", name="Excess Return (obs)"))
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["y_hat"], mode="lines", name="Linha CAPM (ajuste)"))
            fig.update_layout(
                title="CAPM: Benchmark excess vs Portfolio excess",
                xaxis_title="Benchmark excess",
                yaxis_title="Portfolio excess"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scatter CAPM não contém colunas esperadas (x, y, y_hat).")
    else:
        st.info("Sem dados de scatter CAPM.")
