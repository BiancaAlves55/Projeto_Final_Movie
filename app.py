import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ===============================
# CONFIGURAÇÃO
# ===============================
st.set_page_config(page_title="Projeto Final - MovieScope", layout="wide")

# Carregar dataset
df = pd.read_csv("data/tmdb_5000_movies.csv")

# Título do projeto (fixo no topo)
st.title("🎬 Projeto Final - MovieScope")

st.markdown("""
## Desafio
Você foi contratada(o) como **Cientista de Dados Júnior** por uma empresa de análise de performance de streaming chamada **MovieScope**.  

Sua missão é:
- Analisar dados de filmes disponíveis em plataformas digitais.
- Identificar **padrões de sucesso**.
- Entender as **características que influenciam a nota dos filmes**.
- Criar um **modelo de previsão de avaliação** com base em dados históricos.
""")

# Prévia do dataset
st.subheader("Prévia do Dataset")
st.dataframe(df.head())

# ===============================
# MENU LATERAL
# ===============================
opcao = st.sidebar.radio(
    "Escolha uma etapa do projeto:",
    ["Cenário", "Perguntas", "Análises", "Modelos", "Conclusões", "Sugestões de negócio"]
)

# ===============================
# CONTEÚDO POR ABA
# ===============================
if opcao == "Cenário":
    st.markdown("🌍 Cenário")
    st.write("""
    O mercado de streaming é altamente competitivo, e entender os fatores que
    influenciam o sucesso de um filme é essencial para direcionar investimentos
    e estratégias de marketing.
    """)

# -------- ANÁLISES --------

elif opcao == "Análises":
    st.header("📊 Exploração de Dados")

    # -------- FILTROS --------
    st.sidebar.markdown("### 🔎 Filtros")
    min_budget, max_budget = st.sidebar.slider(
        "Orçamento (budget)",
        float(df["budget"].min()),
        float(df["budget"].max()),
        (float(df["budget"].min()), float(df["budget"].max()))
    )

    min_revenue, max_revenue = st.sidebar.slider(
        "Receita (revenue)",
        float(df["revenue"].min()),
        float(df["revenue"].max()),
        (float(df["revenue"].min()), float(df["revenue"].max()))
    )

    min_pop, max_pop = st.sidebar.slider(
        "Popularidade (popularity)",
        float(df["popularity"].min()),
        float(df["popularity"].max()),
        (float(df["popularity"].min()), float(df["popularity"].max()))
    )

    min_votes, max_votes = st.sidebar.slider(
        "Número de votos (vote_count)",
        float(df["vote_count"].min()),
        float(df["vote_count"].max()),
        (float(df["vote_count"].min()), float(df["vote_count"].max()))
    )

    min_score, max_score = st.sidebar.slider(
        "Nota média (vote_average)",
        float(df["vote_average"].min()),
        float(df["vote_average"].max()),
        (float(df["vote_average"].min()), float(df["vote_average"].max()))
    )

    # -------- APLICAR FILTROS --------
    df_filtrado = df[
        (df["budget"] >= min_budget) & (df["budget"] <= max_budget) &
        (df["revenue"] >= min_revenue) & (df["revenue"] <= max_revenue) &
        (df["popularity"] >= min_pop) & (df["popularity"] <= max_pop) &
        (df["vote_count"] >= min_votes) & (df["vote_count"] <= max_votes) &
        (df["vote_average"] >= min_score) & (df["vote_average"] <= max_score)
    ]

    st.write(f"📌 {len(df_filtrado)} filmes selecionados após aplicar os filtros.")

    # -------- GRÁFICOS --------
    st.subheader("🔹 Dispersão: Receita vs Orçamento")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_filtrado, x="budget", y="revenue", hue="vote_average", palette="viridis", ax=ax)
    ax.set_title("Receita x Orçamento (cor = nota média)")
    st.pyplot(fig)

    st.subheader("🔹 Histograma: Distribuição das Notas Médias")
    fig, ax = plt.subplots()
    sns.histplot(df_filtrado["vote_average"], bins=20, kde=True, ax=ax, color="blue")
    ax.set_title("Distribuição da Nota Média (vote_average)")
    st.pyplot(fig)

    st.subheader("🔹 Histograma: Popularidade")
    fig, ax = plt.subplots()
    sns.histplot(df_filtrado["popularity"], bins=20, kde=True, ax=ax, color="green")
    ax.set_title("Distribuição da Popularidade")
    st.pyplot(fig)

elif opcao == "Perguntas":
    st.markdown("❓ Perguntas de Negócio")
    st.write("""
    Algumas perguntas que orientam nossa análise:
    - Quais características estão associadas às maiores notas médias?
    - Filmes com maiores orçamentos realmente têm melhores avaliações?
    - Existe relação entre popularidade e receita?
    - O número de votos influencia diretamente a nota final?
    """)


elif opcao == "Modelos":
    st.markdown("## 🤖 Modelagem Preditiva")

       # Selecionar variáveis
    X = df[["budget", "revenue", "popularity", "vote_count"]]
    y = df["vote_average"]

    # Tratar valores ausentes (caso existam)
    X = X.fillna(0)
    y = y.fillna(0)

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Criar e treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Fazer previsões
    y_pred = modelo.predict(X_test)

    # Avaliação
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Avaliação do Modelo")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    # Comparar previsões reais vs preditas
    st.subheader("🎥 Amostra de Previsões")
    resultado = pd.DataFrame({
        "Real": y_test.values[:20],
        "Previsto": y_pred[:20]
    })
    st.dataframe(resultado)

    # Mostrar coeficientes
    st.subheader("⚖️ Importância das Variáveis")
    coeficientes = pd.DataFrame({
        "Variável": X.columns,
        "Coeficiente": modelo.coef_
    })
    st.dataframe(coeficientes)

    # ===============================
    # VISUALIZAÇÕES
    # ===============================

    # 1. Scatterplot Reais vs Previstos
    st.subheader("📊 Dispersão: Valores Reais vs Previstos")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # linha ideal
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")
    st.pyplot(fig)

    # 2. Linha comparativa
    st.subheader("📉 Comparação em Linha (primeiros 50 filmes de teste)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.values[:50], label="Real", marker="o")
    ax.plot(y_pred[:50], label="Previsto", marker="x")
    ax.legend()
    ax.set_title("Real vs Previsto - 50 primeiros filmes")
    st.pyplot(fig)

elif opcao == "Conclusões":
    st.markdown("## ✅ Conclusões do Projeto")

    st.write("""
    A análise dos dados de filmes da MovieScope permitiu identificar **padrões relevantes de desempenho** e fatores que influenciam diretamente a avaliação média dos títulos.

    ### Principais descobertas:
    - **Popularidade e número de votos** apresentam forte relação com a nota média, indicando que filmes mais discutidos e engajados na plataforma tendem a alcançar melhores avaliações.
    - **Orçamento e receita**, embora importantes, não se mostraram determinantes isoladamente para prever o sucesso — reforçando que grandes investimentos não garantem boas notas.
    - O modelo de **Regressão Linear** desenvolvido apresentou desempenho consistente, explicando parte significativa da variação das notas (R² próximo de 0,7).
  """)


elif opcao == "Sugestões de negócio":
    st.markdown("## 💡 Sugestões de negócio:")
    st.write("""
    - **Curadoria de catálogo:** títulos com alta popularidade e engajamento devem receber maior destaque em recomendações da plataforma.
    - **Marketing direcionado:** campanhas promocionais podem ser mais eficazes quando alinhadas a filmes com potencial de gerar votos e discussões.
    - **Gestão de portfólio:** a receita sozinha não deve ser critério único de seleção; métricas de popularidade e engajamento são indicadores mais robustos do potencial de sucesso.
    """)

    
