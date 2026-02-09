# Ações Analytics (FastAPI + Streamlit)

## Instalação
```bash
pip install -r requirements.txt
```

## Rodar API
```bash
uvicorn app.api:app --reload
```

## Rodar Streamlit
Em outro terminal:
```bash
streamlit run streamlit_app.py
```

## Módulos
1. Retorno diário (simples e log)
2. Retorno anual (anualizado)
3. Sharpe anual
4. Markowitz (simulação + max Sharpe + min variância)
5. CAPM do portfólio (alpha anual, beta, R²)
