import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import date, timedelta

# ✅ pega do env e remove barra final (evita //returns)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Análise de Ações", layout="wide")
st.title("Análise Quantitativa de Ações (FastAPI + Streamlit)")

# Debug opcional (recomendo manter pelo menos enquanto estabiliza deploy)
st.caption(f"API_URL em uso: {API_URL}")

# -----------------------------
# Inputs
# -----------------------------
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
    rf_annual = st.number_input("RF anual (ex: 0.10)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
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

# -----------------------------
# Helpers
# -----------------------------
def call(endpoint, payload):
    url = f"{API_URL}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        st.error(f"Falha ao conectar na API: {url}")
        st.exception(e)
        raise

    if r.status_code != 200:
        st.error(f"Erro na API ({r.status_code}) em {url}")
        # normalmente vem {"detail":"..."} ou HTML; mostramos tudo:
        st.code(r.text)
        raise RuntimeError(r.text)

    return r.json()

def df_from_dict(d: dict) -> pd.DataFrame:
    # ✅ padrão que resolve a maioria dos dicts do FastAPI .to_dict()
    return pd.DataFrame.from_dict(d, orient="index")

# -----------------------------
# Run
# -----------------------------
if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    st.subheader("1) Retorno diário + 2) Retorno anual")

    ret = call("/returns", payload)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])

    # ✅ anual vem como dict {coluna -> {ticker -> valor}} ou similar
    annual = df_from_dict(ret["annual_returns"])
    # garantir ordem/nomes:
    annual.index.name = "ticker"

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

    # ✅ gráfico de barras em formato "long" (sem .T confuso)
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

    st.divider()
    st.subheader("3) Sharpe")

    sh = call("/sharpe", payload)

    sharpe_df = df_from_dict(sh["sharpe_annual"])
    sharpe_df.index.name = "ticker"

    stats_df = df_from_dict(sh["stats_annual"])
    stats_df.index.name = "ticker"

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Dataset: Sharpe anual**")
        st.dataframe(sharpe_df, use_container_width=True)

        sharpe_plot = sharpe_df.reset_index().rename(columns={"sharpe_annual": "value"})
        fig = px.bar(sharpe_plot, x="ticker", y="value", title="Sharpe anual")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("**Dataset: Stats anual (ret/vol)**")
        st.dataframe(stats_df, use_container_width=True)

        fig = px.scatter(
            stats_df.reset_index(),
            x="vol_annual",
            y="ret_annual",
            title="Risco x Retorno (anual)",
            text="ticker"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("4) Markowitz")

    if len(tickers) < 2:
        st.info("Markowitz requer 2+ tickers. Pulei este módulo.")
    else:
        mk = call("/markowitz", payload)
        pts = pd.DataFrame(mk["frontier_points"])

        max_sh = mk["max_sharpe"]
        min_v = mk["min_variance"]

        c5, c6 = st.columns([2, 1])
        with c5:
            fig = px.scatter(
                pts,
                x="vol",
                y="ret",
                title="Simulação Markowitz (vol x ret)",
                hover_data=["sharpe"]
            )
            fig.add_trace(go.Scatter(
                x=[max_sh["vol"]],
                y=[max_sh["ret"]],
                mode="markers",
                name="Max Sharpe"
            ))
            fig.add_trace(go.Scatter(
                x=[min_v["vol"]],
                y=[pts.loc[pts["vol"].idxmin(), "ret"]],
                mode="markers",
                name="Min Variância"
            ))
            st.plotly_chart(fig, use_container_width=True)

        with c6:
            st.markdown("**Max Sharpe**")
            st.json(max_sh)
            st.markdown("**Min Variância**")
            st.json(min_v)

    st.divider()
    st.subheader("5) CAPM do Portfólio (alpha, beta, R²)")

    capm = call("/capm", payload)
    st.json(capm)
