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

# Debug útil em produção
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
    rf_annual = st.number_input(
        "RF anual (ex: 0.10)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )
with col5:
    # ✅ limite para evitar explosão de CPU no free tier
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

# -----------------------------
# Helpers
# -----------------------------
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

def df_from_dict_index(d: dict) -> pd.DataFrame:
    """Converte dict -> DataFrame com orientação por índice (tickers no index) quando fizer sentido."""
    if d is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame.from_dict(d, orient="index")
    except Exception:
        # fallback
        return pd.DataFrame(d)

def normalize_single_metric(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Garante formato: index=ticker e coluna=metric_name
    Cobre formatos comuns vindos de pandas .to_dict() em diferentes orientações.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=[metric_name])
        out.index.name = "ticker"
        return out

    # Caso "bom": já existe a coluna metric_name
    if metric_name in df.columns:
        out = df.copy()

    # Caso "invertido": metric_name veio no index (e tickers como colunas)
    elif metric_name in df.index:
        out = df.T.copy()
        # se ficou 1 coluna sem nome ou nomes estranhos, força:
        if out.shape[1] == 1:
            out.columns = [metric_name]

    # Caso: veio 1 coluna com nome diferente
    elif df.shape[1] == 1:
        out = df.copy()
        out.columns = [metric_name]

    else:
        # Fallback: tenta transpor
        out = df.T.copy()
        if out.shape[1] == 1:
            out.columns = [metric_name]

    out.index.name = "ticker"
    return out

def normalize_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera colunas: ret_annual, vol_annual; index: ticker
    Se vier invertido (essas chaves no index), transpõe.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=["ret_annual", "vol_annual"])
        out.index.name = "ticker"
        return out

    if ("ret_annual" in df.index) or ("vol_annual" in df.index):
        out = df.T.copy()
    else:
        out = df.copy()

    out.index.name = "ticker"
    return out

# -----------------------------
# Run
# -----------------------------
if run:
    if len(tickers) == 0:
        st.warning("Informe ao menos 1 ticker.")
        st.stop()

    # =============================
    # 1) Retorno diário + 2) anual
    # =============================
    st.subheader("1) Retorno diário + 2) Retorno anual")
    ret = call("/returns", payload)

    daily_simple = pd.DataFrame(ret["daily_returns"]["simple"])
    daily_log = pd.DataFrame(ret["daily_returns"]["log"])

    # annual_returns vem como dict; normalizar robustamente
    raw_annual = pd.DataFrame(ret["annual_returns"])
    annual = normalize_stats(raw_annual)

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

    # barras em formato "long" (evita .T confuso)
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

    # =============================
    # 3) Sharpe
    # =============================
    st.divider()
    st.subheader("3) Sharpe")

    sh = call("/sharpe", payload)

    # sharpe_annual e stats_annual podem vir em orientações diferentes
    raw_sharpe = pd.DataFrame(sh["sharpe_annual"])
    raw_stats = pd.DataFrame(sh["stats_annual"])

    sharpe_df = normalize_single_metric(raw_sharpe, "sharpe_annual")
    stats_df = normalize_stats(raw_stats)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Dataset: Sharpe anual**")
        st.dataframe(sharpe_df, use_container_width=True)

        if (not sharpe_df.empty) and ("sharpe_annual" in sharpe_df.columns):
            fig = px.bar(
                sharpe_df.reset_index(),
                x="ticker",
                y="sharpe_annual",
                title="Sharpe anual"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sharpe retornou vazio ou sem coluna esperada.")

    with c4:
        st.markdown("**Dataset: Stats anual (ret/vol)**")
        st.dataframe(stats_df, use_container_width=True)

        if (
            (not stats_df.empty)
            and ("vol_annual" in stats_df.columns)
            and ("ret_annual" in stats_df.columns)
        ):
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

    # =============================
    # 4) Markowitz
    # =============================
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
            # pon
