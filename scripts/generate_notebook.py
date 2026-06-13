import json
import os

def build_notebook():
    cells = []
    
    # ----------------------------------------------------
    # Title & Authors
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Reconhecimento de Padrões - Trabalho 2\n",
            "## Análise Comparativa de Métodos de Regressão\n",
            "\n",
            "**Autores:**\n",
            "- Alan P. Berger Saar\n",
            "- Leonardo Herkenhoff\n",
            "\n",
            "**Professor:** Dr. Sérgio Nery Simões\n",
            "**Instituição:** PPCOMP - IFES\n",
            "\n",
            "---"
        ]
    })
    
    # ----------------------------------------------------
    # Introduction
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Introdução e Objetivo\n",
            "\n",
            "Este notebook apresenta a implementação completa e detalhada do pipeline experimental para o **Trabalho 2 (Regressão)**.\n",
            "Nosso objetivo é prever a variável-alvo `cnt` (contagem horária de aluguéis de bicicletas) a partir de variáveis temporais e meteorológicas usando o dataset *Bike Sharing* da UCI.\n",
            "\n",
            "O pipeline é composto por 5 etapas:\n",
            "1. **Análise Exploratória dos Dados (EDA)**\n",
            "2. **Pré-processamento**\n",
            "3. **Ajuste de Modelos de Regressão** (OLS, Ridge, Lasso, Random Forest e XGBoost)\n",
            "4. **Avaliação Experimental** (Cross-Validation de 10 Folds e Hold-Out no Teste)\n",
            "5. **Diagnóstico dos Resíduos** para os dois melhores modelos\n",
            "\n",
            "Abaixo, importamos as bibliotecas necessárias para começar."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import scipy.stats as stats\n",
            "import joblib\n",
            "\n",
            "from ucimlrepo import fetch_ucirepo\n",
            "\n",
            "from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
            "from sklearn.ensemble import RandomForestRegressor\n",
            "from xgboost import XGBRegressor\n",
            "\n",
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
            "\n",
            "# Configurações estéticas de gráficos\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams.update({\n",
            "    'font.size': 12,\n",
            "    'axes.labelsize': 12,\n",
            "    'axes.titlesize': 14,\n",
            "    'xtick.labelsize': 10,\n",
            "    'ytick.labelsize': 10,\n",
            "    'figure.titlesize': 16\n",
            "})\n",
            "\n",
            "# Garantir pasta para as figuras do relatório\n",
            "os.makedirs(\"../relatorio/img\", exist_ok=True)\n",
            "print(\"Bibliotecas importadas e diretório de imagens verificado.\")"
        ]
    })
    
    # ----------------------------------------------------
    # Etapa 1 - EDA
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Etapa 1 — Análise Exploratória dos Dados (EDA)\n",
            "\n",
            "Antes de treinar qualquer modelo, precisamos entender os nossos dados. Vamos carregar o dataset diretamente do repositório da UCI e realizar análises visuais e estatísticas.\n",
            "\n",
            "### Perguntas que a EDA responderá:\n",
            "1. **Qual a distribuição da variável-alvo `cnt`?** Ela é simétrica ou possui assimetria?\n",
            "2. **Por que a transformação logarítmica (`log1p`) é recomendada?**\n",
            "3. **Quais características apresentam maior correlação com `cnt`?**\n",
            "4. **Existe multicolinearidade?** (Por exemplo, entre `temp` e `atemp`)\n",
            "5. **Como o consumo varia ao longo do dia, da semana e das estações do ano?**\n",
            "6. **Quais são os impactos do clima (`weathersit`) e dias úteis (`workingday`) na demanda?**"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Carregando o dataset hourly (ID 275)\n",
            "print(\"Buscando dataset do repositório UCI...\")\n",
            "dataset = fetch_ucirepo(id=275)\n",
            "\n",
            "X = dataset.data.features.drop(columns=['dteday'])\n",
            "y = dataset.data.targets['cnt']\n",
            "\n",
            "# Criando dataframe único para EDA\n",
            "df = X.copy()\n",
            "df['cnt'] = y\n",
            "\n",
            "print(f\"Dataset carregado. Dimensões: {df.shape}\")\n",
            "df.info()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.1 Distribuição da Variável-Alvo `cnt`\n",
            "\n",
            "A variável-alvo `cnt` é uma variável de contagem. Em problemas do mundo real, contagens frequentemente apresentam distribuições com cauda longa à direita (**assimetria positiva**).\n",
            "\n",
            "#### Por que isso é um problema para modelos lineares?\n",
            "Modelos lineares assumem que os erros (resíduos) são normalmente distribuídos e homocedásticos (variância constante). Quando a variável-alvo tem assimetria severa e alta variabilidade dependente de sua escala, essas premissas são violadas, fazendo com que o modelo erre mais em valores extremos.\n",
            "\n",
            "#### A Transformação Logarítmica (`log1p`):\n",
            "A transformação $y \\to \\log(1 + y)$ (conhecida como `log1p`):\n",
            "1. Estabiliza a variância (mitiga heterocedasticidade).\n",
            "2. Comprime a cauda longa de valores altos, tornando a distribuição aproximadamente normal.\n",
            "3. O \"+1\" garante estabilidade numérica caso existam contagens iguais a 0 (embora no dataset o mínimo de `cnt` seja 1)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Estatísticas descritivas de cnt e seu skewness\n",
            "print(df['cnt'].describe())\n",
            "print(f\"Assimetria (skewness) da variável original: {df['cnt'].skew():.3f}\")\n",
            "print(f\"Assimetria (skewness) da variável transformada log1p: {np.log1p(df['cnt']).skew():.3f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot da distribuição da variável-alvo antes e depois da transformação log1p\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "sns.histplot(df['cnt'], bins=50, kde=True, color='steelblue', edgecolor='white', ax=axes[0])\n",
            "axes[0].set_title('Distribuição de cnt (Original)')\n",
            "axes[0].set_xlabel('cnt (Aluguéis de Bicicletas)')\n",
            "axes[0].set_ylabel('Frequência')\n",
            "\n",
            "sns.histplot(np.log1p(df['cnt']), bins=50, kde=True, color='darkorange', edgecolor='white', ax=axes[1])\n",
            "axes[1].set_title('Distribuição de log1p(cnt)')\n",
            "axes[1].set_xlabel('log1p(cnt)')\n",
            "axes[1].set_ylabel('Frequência')\n",
            "\n",
            "plt.tight_layout()\n",
            "fig.savefig(\"../relatorio/img/distribuicao_cnt.png\", dpi=150, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.2 Mapa de Correlações (Heatmap)\n",
            "\n",
            "O coeficiente de correlação de Pearson mede o grau de relação linear entre duas variáveis.\n",
            "\n",
            "#### Multicolinearidade:\n",
            "Ocorre quando duas ou mais características estão fortemente correlacionadas entre si. \n",
            "- Exemplo típico: `temp` (temperatura real) e `atemp` (sensação térmica).\n",
            "- **Problema:** A multicolinearidade infla a variância dos coeficientes em modelos lineares ordinários (OLS), tornando os coeficientes instáveis e difíceis de interpretar. \n",
            "- **Solução:** Modelos de regularização como *Ridge* e *Lasso* ajudam a controlar esse problema, penalizando coeficientes muito grandes."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "sns.heatmap(df.corr(), annot=True, fmt=\".2f\", cmap=\"coolwarm\", center=0, ax=ax, linewidths=0.5)\n",
            "ax.set_title('Mapa de Correlações entre Variáveis')\n",
            "plt.tight_layout()\n",
            "fig.savefig(\"../relatorio/img/heatmap_correlacoes.png\", dpi=150, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.3 Padrões Temporais (Hora, Mês, Estação)\n",
            "\n",
            "A demanda de bicicletas varia fortemente ao longo do tempo por conta das rotinas urbanas e climáticas:\n",
            "- **Hora do dia (`hr`):** Esperamos picos em horários de pico (ida/volta do trabalho, ~8h e ~17h-18h).\n",
            "- **Mês (`mnth`):** Meses mais quentes (meio do ano no hemisfério norte) tendem a apresentar maior consumo.\n",
            "- **Estação (`season`):** Queda acentuada na primavera/inverno e picos no verão/outono."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
            "\n",
            "df.groupby('hr')['cnt'].mean().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')\n",
            "axes[0].set_title('Média de Aluguéis por Hora')\n",
            "axes[0].set_xlabel('Hora do Dia')\n",
            "axes[0].set_ylabel('cnt Médio')\n",
            "axes[0].tick_params(axis='x', rotation=0)\n",
            "\n",
            "df.groupby('mnth')['cnt'].mean().plot(kind='bar', ax=axes[1], color='seagreen', edgecolor='white')\n",
            "axes[1].set_title('Média de Aluguéis por Mês')\n",
            "axes[1].set_xlabel('Mês')\n",
            "axes[1].set_ylabel('cnt Médio')\n",
            "axes[1].tick_params(axis='x', rotation=0)\n",
            "\n",
            "df.groupby('season')['cnt'].mean().plot(kind='bar', ax=axes[2], color='darkorange', edgecolor='white')\n",
            "axes[2].set_title('Média de Aluguéis por Estação')\n",
            "axes[2].set_xlabel('Estação')\n",
            "axes[2].set_ylabel('cnt Médio')\n",
            "axes[2].tick_params(axis='x', rotation=0)\n",
            "axes[2].set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'])\n",
            "\n",
            "plt.tight_layout()\n",
            "fig.savefig(\"../relatorio/img/padroes_temporais.png\", dpi=150, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.4 Condições Climáticas e Contexto do Dia (Fins de semana, clima)\n",
            "\n",
            "Analisamos a variação do consumo de acordo com a condição climática (`weathersit`) e os dias úteis (`workingday`).\n",
            "- **`weathersit`:** 1 = Claro/Parcialmente nublado, 2 = Névoa/Nublado, 3 = Chuva leve/Neve leve, 4 = Chuva intensa/Tempestade.\n",
            "- **`workingday`:** 1 = Dia útil, 0 = Fim de semana ou Feriado."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
            "\n",
            "sns.boxplot(data=df, x='weathersit', y='cnt', ax=axes[0], palette='Blues')\n",
            "axes[0].set_title('cnt por Condição Climática')\n",
            "axes[0].set_xlabel('weathersit (1=Claro → 4=Chuva Forte)')\n",
            "axes[0].set_ylabel('cnt')\n",
            "\n",
            "sns.boxplot(data=df, x='workingday', y='cnt', ax=axes[1], palette='Set1')\n",
            "axes[1].set_title('cnt — Dia Útil vs. Não Útil')\n",
            "axes[1].set_xticklabels(['Fim de Semana/Feriado', 'Dia Útil'])\n",
            "axes[1].set_xlabel('Tipo de Dia')\n",
            "axes[1].set_ylabel('cnt')\n",
            "\n",
            "sns.boxplot(data=df, x='weekday', y='cnt', ax=axes[2], palette='Set2')\n",
            "axes[2].set_title('cnt por Dia da Semana')\n",
            "axes[2].set_xlabel('Dia (0=Domingo → 6=Sábado)')\n",
            "axes[2].set_ylabel('cnt')\n",
            "\n",
            "plt.tight_layout()\n",
            "fig.savefig(\"../relatorio/img/clima_e_contexto.png\", dpi=150, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    })
    
    # ----------------------------------------------------
    # Etapa 2 - Preprocessing
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Etapa 2 — Pré-processamento\n",
            "\n",
            "O pré-processamento prepara os dados de forma robusta e metodológica:\n",
            "1. **Divisão Hold-out:** Separa 80% dos dados para treino e 20% para teste. Fazemos isso de forma **estratificada por estação (`season`)** para garantir que a proporção das estações seja idêntica em ambos os conjuntos.\n",
            "2. **Prevenção de Vazamento de Dados (Data Leakage):** Métricas como média e desvio padrão (`StandardScaler`) devem ser calculadas *apenas* a partir do conjunto de treino e replicadas no conjunto de teste. Para evitar erros humanos, encapsulamos as transformações no `ColumnTransformer` e nos `Pipeline` do scikit-learn.\n",
            "3. **Tipos de Features:**\n",
            "   - **Contínuas (`temp`, `atemp`, `hum`, `windspeed`):** Padronizadas usando `StandardScaler` ($z = \\frac{x - \\mu}{\\sigma}$).\n",
            "   - **Ordinais/Categóricas Numéricas (`hr`, `mnth`, `season`, `weekday`, `weathersit`):** Mantidas como ordinais numéricas simples conforme especificado pelo enunciado.\n",
            "   - **Binárias (`holiday`, `workingday`, `yr`):** Mantidas como estão (valores 0 e 1).\n",
            "4. **Variável-Alvo:** Aplicação automática do `log1p(cnt)` no treino, e reversão com `expm1()` no momento da avaliação."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Separando features e alvo\n",
            "X = df.drop(columns=['cnt'])\n",
            "y = df['cnt']\n",
            "\n",
            "# Split Hold-out (80% treino, 20% teste) estratificado por 'season'\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.20, random_state=42, stratify=X['season']\n",
            ")\n",
            "\n",
            "print(f\"Treino shape: {X_train.shape}, Teste shape: {X_test.shape}\")\n",
            "\n",
            "# Definindo tipos de colunas\n",
            "num_cols = ['temp', 'atemp', 'hum', 'windspeed']\n",
            "cat_ord_cols = ['hr', 'mnth', 'season', 'weekday', 'weathersit']\n",
            "bin_cols = ['holiday', 'workingday', 'yr']\n",
            "\n",
            "# Aplicando transformação log1p na variável alvo\n",
            "y_train_log = np.log1p(y_train)\n",
            "y_test_log = np.log1p(y_test)\n",
            "\n",
            "# Configurando o ColumnTransformer para pré-processamento das features\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', StandardScaler(), num_cols),\n",
            "        ('ord', 'passthrough', cat_ord_cols),\n",
            "        ('bin', 'passthrough', bin_cols)\n",
            "    ]\n",
            ")\n",
            "\n",
            "print(\"Pré-processamento configurado via ColumnTransformer.\")"
        ]
    })
    
    # ----------------------------------------------------
    # Etapa 3 - Models
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Etapa 3 — Modelos de Regressão e Tuning\n",
            "\n",
            "Nesta etapa, implementamos 5 modelos de complexidades crescentes:\n",
            "\n",
            "### 3.1 Conceitos teóricos dos Modelos:\n",
            "\n",
            "1. **OLS (Ordinary Least Squares):**\n",
            "   - **O que é:** O modelo de regressão linear clássico. Minimiza a soma dos erros quadráticos residuais (RSS):\n",
            "     $$RSS = \\sum_{i=1}^n (y_i - (\\beta_0 + \\sum \\beta_j X_{ij}))^2$$\n",
            "   - **Decisão:** Serve como o baseline do nosso trabalho.\n",
            "\n",
            "2. **Regressão Ridge (Regularização L2):**\n",
            "   - **O que é:** Adiciona uma penalidade baseada na norma L2 (soma dos quadrados dos coeficientes) à função de perda:\n",
            "     $$\\text{Perda} = RSS + \\alpha \\sum_{j=1}^p \\beta_j^2$$\n",
            "   - **Por que usar:** Impede que os coeficientes fiquem excessivamente grandes em situações de multicolinearidade (como `temp` e `atemp`), reduzindo a variância do modelo ao custo de introduzir um pequeno viés.\n",
            "\n",
            "3. **Regressão Lasso (Regularização L1):**\n",
            "   - **O que é:** Adiciona uma penalidade baseada na norma L1 (soma dos valores absolutos dos coeficientes):\n",
            "     $$\\text{Perda} = RSS + \\alpha \\sum_{j=1}^p |\\beta_j|$$\n",
            "   - **Por que usar:** Força coeficientes de variáveis irrelevantes ou redundantes a serem **exatamente zero**. Funciona como um método embutido de seleção de características (*feature selection*).\n",
            "\n",
            "4. **Random Forest Regressor:**\n",
            "   - **O que é:** Um algoritmo de *ensemble* baseado em *Bagging* (Bootstrap Aggregating). Treina múltiplas árvores de decisão independentes em amostras aleatórias dos dados e tira a média das previsões.\n",
            "   - **Por que usar:** Excelente para modelar padrões altamente não-lineares e interações complexas entre variáveis, sem sofrer tanto com overfitting como árvores individuais.\n",
            "\n",
            "5. **XGBoost Regressor (Gradient Boosting):**\n",
            "   - **O que é:** Um algoritmo baseado em *Boosting*. Ao contrário do Bagging, as árvores são construídas de forma sequencial. Cada nova árvore é treinada para corrigir os erros (resíduos) cometidos pelas árvores anteriores, descendo o gradiente da função de perda.\n",
            "   - **Por que usar:** O XGBoost é extremamente otimizado, inclui regularização interna em sua função objetiva, suporta execução paralela e costuma produzir o melhor desempenho preditivo em dados tabulares.\n",
            "\n",
            "### 3.2 Protocolo de Tuning de Hiperparâmetros:\n",
            "Usaremos o **`RandomizedSearchCV`** com **validação cruzada de 5 folds** sobre o conjunto de treino ($n_{\\text{iter}} \\ge 30$). Isso nos permite explorar o espaço hiperparamétrico de forma computacionalmente viável."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Dicionário para guardar os melhores modelos e hiperparâmetros\n",
            "best_estimators = {}\n",
            "\n",
            "# Configurando validação cruzada para o tuning\n",
            "cv_tuning = KFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "print(\"Configuração inicial de Tuning pronta.\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 1. Ajuste OLS (Ordinary Least Squares)\n",
            "O OLS clássico não possui hiperparâmetros de regularização a serem ajustados."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ols_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('regressor', LinearRegression())\n",
            "])\n",
            "\n",
            "# Fit OLS no treino completo\n",
            "ols_pipeline.fit(X_train, y_train_log)\n",
            "best_estimators['OLS'] = ols_pipeline\n",
            "print(\"Modelo OLS ajustado com sucesso.\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 2. Ajuste Ridge\n",
            "Hiperparâmetro a ser ajustado: $\\alpha$ (intensidade da penalização L2)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ridge_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('regressor', Ridge())\n",
            "])\n",
            "\n",
            "ridge_param_dist = {\n",
            "    'regressor__alpha': stats.loguniform(1e-3, 1e3)  # Espaço log-contínuo\n",
            "}\n",
            "\n",
            "ridge_search = RandomizedSearchCV(\n",
            "    ridge_pipeline, param_distributions=ridge_param_dist, \n",
            "    n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', \n",
            "    random_state=42, n_jobs=-1\n",
            ")\n",
            "\n",
            "ridge_search.fit(X_train, y_train_log)\n",
            "best_estimators['Ridge'] = ridge_search.best_estimator_\n",
            "print(f\"Ridge melhor alpha: {ridge_search.best_params_['regressor__alpha']:.4f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 3. Ajuste Lasso\n",
            "Hiperparâmetro a ser ajustado: $\\alpha$ (intensidade da penalização L1). O Lasso pode zerar coeficientes."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lasso_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('regressor', Lasso(max_iter=10000))\n",
            "])\n",
            "\n",
            "lasso_param_dist = {\n",
            "    'regressor__alpha': stats.loguniform(1e-5, 1.0)\n",
            "}\n",
            "\n",
            "lasso_search = RandomizedSearchCV(\n",
            "    lasso_pipeline, param_distributions=lasso_param_dist, \n",
            "    n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', \n",
            "    random_state=42, n_jobs=-1\n",
            ")\n",
            "\n",
            "lasso_search.fit(X_train, y_train_log)\n",
            "best_estimators['Lasso'] = lasso_search.best_estimator_\n",
            "print(f\"Lasso melhor alpha: {lasso_search.best_params_['regressor__alpha']:.6f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 4. Ajuste Random Forest\n",
            "Ajustamos a profundidade das árvores, número mínimo de amostras por folha e número de estimadores."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('regressor', RandomForestRegressor(random_state=42))\n",
            "])\n",
            "\n",
            "rf_param_dist = {\n",
            "    'regressor__n_estimators': stats.randint(50, 200),\n",
            "    'regressor__max_depth': stats.randint(8, 25),\n",
            "    'regressor__min_samples_split': stats.randint(2, 12),\n",
            "    'regressor__max_features': ['sqrt', 'log2', None]\n",
            "}\n",
            "\n",
            "rf_search = RandomizedSearchCV(\n",
            "    rf_pipeline, param_distributions=rf_param_dist, \n",
            "    n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', \n",
            "    random_state=42, n_jobs=-1\n",
            ")\n",
            "\n",
            "print(\"Ajustando hiperparâmetros da Random Forest (isso pode levar alguns minutos)...\")\n",
            "rf_search.fit(X_train, y_train_log)\n",
            "best_estimators['Random Forest'] = rf_search.best_estimator_\n",
            "print(f\"Random Forest melhores parâmetros: {rf_search.best_params_}\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### 5. Ajuste XGBoost\n",
            "Ajustamos taxa de aprendizado (`learning_rate`), profundidade máxima (`max_depth`), número de árvores (`n_estimators`), regularização L1 (`alpha`) e L2 (`lambda`)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "xgb_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('regressor', XGBRegressor(random_state=42))\n",
            "])\n",
            "\n",
            "xgb_param_dist = {\n",
            "    'regressor__n_estimators': stats.randint(80, 250),\n",
            "    'regressor__max_depth': stats.randint(4, 10),\n",
            "    'regressor__learning_rate': stats.uniform(0.01, 0.25),\n",
            "    'regressor__subsample': stats.uniform(0.6, 0.4),\n",
            "    'regressor__colsample_bytree': stats.uniform(0.6, 0.4),\n",
            "    'regressor__reg_alpha': stats.loguniform(1e-3, 10.0),\n",
            "    'regressor__reg_lambda': stats.loguniform(1e-3, 10.0)\n",
            "}\n",
            "\n",
            "xgb_search = RandomizedSearchCV(\n",
            "    xgb_pipeline, param_distributions=xgb_param_dist, \n",
            "    n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', \n",
            "    random_state=42, n_jobs=-1\n",
            ")\n",
            "\n",
            "print(\"Ajustando hiperparâmetros do XGBoost...\")\n",
            "xgb_search.fit(X_train, y_train_log)\n",
            "best_estimators['XGBoost'] = xgb_search.best_estimator_\n",
            "print(f\"XGBoost melhores parâmetros: {xgb_search.best_params_}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Salvar os modelos ajustados usando joblib para evitar retreino\n",
            "os.makedirs(\"../modelos_salvos\", exist_ok=True)\n",
            "for name, estimator in best_estimators.items():\n",
            "    joblib.dump(estimator, f\"../modelos_salvos/{name.lower().replace(' ', '_')}_model.pkl\")\n",
            "print(\"Todos os melhores estimadores foram persistidos localmente.\")"
        ]
    })
    
    # ----------------------------------------------------
    # Etapa 4 - Evaluation
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Etapa 4 — Avaliação Experimental\n",
            "\n",
            "Com as melhores configurações de hiperparâmetros fixadas, faremos uma avaliação metodológica rigorosa:\n",
            "\n",
            "1. **10-Fold Cross-Validation (CV):**\n",
            "   - Avalia a variabilidade e estabilidade dos modelos no conjunto de treino completo usando 10 divisões distintas.\n",
            "\n",
            "2. **Avaliação Final no Teste (Hold-out):**\n",
            "   - Mede a capacidade de generalização de cada modelo em dados nunca vistos.\n",
            "\n",
            "### Métricas de Avaliação:\n",
            "Como o treino foi realizado no espaço transformado (`log1p`), **todas as previsões devem ser revertidas para o espaço original (bicicletas/hora) usando `expm1`** antes de calcular as métricas:\n",
            "\n",
            "- **RMSE (Root Mean Squared Error):** Penaliza erros de grande magnitude (ótimo para detectar erros discrepantes).\n",
            "- **MAE (Mean Absolute Error):** Média dos erros absolutos. Mais robusto contra outliers.\n",
            "- **R² (Coeficiente de Determinação):** Proporção da variância total explicada pelo modelo.\n",
            "- **MAPE (Mean Absolute Percentage Error):** Mede o erro percentual médio. Intuitivo para apresentação de negócios."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def calculate_metrics(y_true, y_pred):\n",
            "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
            "    mae = mean_absolute_error(y_true, y_pred))\n",
            "    r2 = r2_score(y_true, y_pred)\n",
            "    # Evitar divisão por zero no MAPE\n",
            "    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100\n",
            "    return rmse, mae, r2, mape\n",
            "\n",
            "# Inicializar tabelas de resultados\n",
            "cv_results = []\n",
            "holdout_results = []\n",
            "\n",
            "cv_10fold = KFold(n_splits=10, shuffle=True, random_state=42)\n",
            "\n",
            "print(\"Funções e parâmetros de avaliação inicializados.\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for name, pipeline in best_estimators.items():\n",
            "    print(f\"Avaliando {name} via 10-Fold CV...\")\n",
            "    \n",
            "    # Listas para guardar as métricas de cada fold da CV (no espaço original)\n",
            "    rmse_folds, mae_folds, r2_folds, mape_folds = [], [], [], []\n",
            "    \n",
            "    # Loop manual de CV para reverter a transformação e extrair métricas corretas\n",
            "    for train_idx, val_idx in cv_10fold.split(X_train):\n",
            "        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]\n",
            "        y_tr_log, y_val = y_train_log.iloc[train_idx], y_train.iloc[val_idx] # y_val em escala original!\n",
            "        \n",
            "        # Clonar e treinar o pipeline\n",
            "        from sklearn.base import clone\n",
            "        fold_pipeline = clone(pipeline)\n",
            "        fold_pipeline.fit(X_tr, y_tr_log)\n",
            "        \n",
            "        # Previsão no fold de validação e conversão expm1\n",
            "        y_val_pred_log = fold_pipeline.predict(X_val)\n",
            "        y_val_pred = np.expm1(y_val_pred_log)\n",
            "        \n",
            "        rmse_f, mae_f, r2_f, mape_f = calculate_metrics(y_val, y_val_pred)\n",
            "        rmse_folds.append(rmse_f)\n",
            "        mae_folds.append(mae_f)\n",
            "        r2_folds.append(r2_f)\n",
            "        mape_folds.append(mape_f)\n",
            "        \n",
            "    cv_results.append({\n",
            "        'Modelo': name,\n",
            "        'RMSE CV': f\"{np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}\",\n",
            "        'MAE CV': f\"{np.mean(mae_folds):.2f} ± {np.std(mae_folds):.2f}\",\n",
            "        'R2 CV': f\"{np.mean(r2_folds):.3f} ± {np.std(r2_folds):.3f}\",\n",
            "        'MAPE CV (%)': f\"{np.mean(mape_folds):.2f}% ± {np.std(mape_folds):.2f}%\"\n",
            "    })\n",
            "    \n",
            "    # --- Avaliação Hold-out no Teste ---\n",
            "    # O pipeline completo já foi treinado no X_train inteiro na Etapa 3\n",
            "    y_test_pred_log = pipeline.predict(X_test)\n",
            "    y_test_pred = np.expm1(y_test_pred_log)\n",
            "    \n",
            "    rmse_t, mae_t, r2_t, mape_t = calculate_metrics(y_test, y_test_pred)\n",
            "    holdout_results.append({\n",
            "        'Modelo': name,\n",
            "        'RMSE Teste': rmse_t,\n",
            "        'MAE Teste': mae_t,\n",
            "        'R2 Teste': r2_t,\n",
            "        'MAPE Teste (%)': mape_t\n",
            "    })\n",
            "\n",
            "df_cv = pd.DataFrame(cv_results)\n",
            "df_holdout = pd.DataFrame(holdout_results).sort_values(by='RMSE Teste')\n",
            "\n",
            "print(\"Avaliação finalizada!\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 Tabela Comparativa de Resultados\n",
            "\n",
            "Vamos exibir e salvar as tabelas de métricas comparativas ordenadas pelo RMSE no conjunto de teste."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== Resultados da Validação Cruzada (10-Fold CV) ===\")\n",
            "print(df_cv.to_string(index=False))\n",
            "print(\"\\n=== Resultados do Conjunto de Teste (Hold-out) ===\")\n",
            "print(df_holdout.to_string(index=False))\n",
            "\n",
            "# Salvar tabelas para inclusão no relatório\n",
            "df_cv.to_csv(\"../relatorio/img/resultado_cv.csv\", index=False)\n",
            "df_holdout.to_csv(\"../relatorio/img/resultado_holdout.csv\", index=False)"
        ]
    })
    
    # ----------------------------------------------------
    # Etapa 5 - Residuals
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Etapa 5 — Diagnóstico dos Resíduos\n",
            "\n",
            "A análise de resíduos é um dos passos fundamentais para validar o ajuste de modelos de machine learning.\n",
            "\n",
            "### O que são os Resíduos?\n",
            "O resíduo é a diferença entre o valor real e o valor predito pelo modelo: $e_i = y_i - \\hat{y}_i$. Faremos essa análise no **espaço original** de bicicletas/hora.\n",
            "\n",
            "### O que buscamos na Análise?\n",
            "1. **Distribuição dos Resíduos:** Devem ser aproximadamente normais, centrados em zero. Desvios indicam que o modelo erra sistematicamente em direções específicas.\n",
            "2. **Resíduos vs. Preditos:** Gráfico de dispersão entre $e_i$ e $\\hat{y}_i$. Não deve exibir padrões geométricos (como formato de funil, indicando heterocedasticidade).\n",
            "3. **Q-Q Plot:** Plota os quantis teóricos da normal vs. quantis dos resíduos. Desvios nas caudas mostram caudas pesadas ou outliers severos.\n",
            "4. **Resíduos vs. Features (`hr`, `temp`, `season`):** Verifica se o erro do modelo varia conforme faixas de certas variáveis. Revela se o modelo falhou em aprender padrões não-lineares.\n",
            "5. **Teste Formal de Normalidade:** D'Agostino-Pearson (`scipy.stats.normaltest`) testa a hipótese nula de que os resíduos vêm de uma distribuição normal.\n",
            "\n",
            "Identificamos abaixo os **dois melhores modelos** (menores RMSE no Hold-out) para aplicar o diagnóstico."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Obter os dois melhores modelos a partir do ranking do holdout\n",
            "top_2_names = df_holdout['Modelo'].head(2).tolist()\n",
            "print(f\"Os dois melhores modelos identificados foram: {top_2_names}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for model_name in top_2_names:\n",
            "    pipeline = best_estimators[model_name]\n",
            "    y_pred_log = pipeline.predict(X_test)\n",
            "    y_pred = np.expm1(y_pred_log)\n",
            "    \n",
            "    # Resíduos no espaço original\n",
            "    residuals = y_test - y_pred\n",
            "    \n",
            "    # --- 1. Histograma + Curva KDE dos Resíduos ---\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "    fig.suptitle(f\"Análise de Resíduos: {model_name}\", fontsize=16)\n",
            "    \n",
            "    sns.histplot(residuals, kde=True, bins=50, color='darkred', edgecolor='white', ax=axes[0, 0])\n",
            "    axes[0, 0].set_title(\"Distribuição dos Resíduos\")\n",
            "    axes[0, 0].set_xlabel(\"Resíduo (Real - Predito)\")\n",
            "    \n",
            "    # --- 2. Resíduos vs Valores Preditos ---\n",
            "    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, color='purple', edgecolors='none')\n",
            "    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)\n",
            "    axes[0, 1].set_title(\"Resíduos vs. Valores Preditos\")\n",
            "    axes[0, 1].set_xlabel(\"Valores Preditos\")\n",
            "    axes[0, 1].set_ylabel(\"Resíduo\")\n",
            "    \n",
            "    # --- 3. Q-Q Plot ---\n",
            "    stats.probplot(residuals, dist=\"norm\", plot=axes[1, 0])\n",
            "    axes[1, 0].set_title(\"Q-Q Plot\")\n",
            "    \n",
            "    # --- 4. Resíduos vs Feature (Temperatura) ---\n",
            "    axes[1, 1].scatter(X_test['temp'], residuals, alpha=0.3, color='teal', edgecolors='none')\n",
            "    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)\n",
            "    axes[1, 1].set_title(\"Resíduos vs. Temperatura (Normalizada)\")\n",
            "    axes[1, 1].set_xlabel(\"Temperatura\")\n",
            "    axes[1, 1].set_ylabel(\"Resíduo\")\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    filename = f\"residuos_{model_name.lower().replace(' ', '_')}.png\"\n",
            "    fig.savefig(f\"../relatorio/img/{filename}\", dpi=150, bbox_inches=\"tight\")\n",
            "    plt.show()\n",
            "    \n",
            "    # --- Teste Formal de Normalidade ---\n",
            "    stat, p_val = stats.normaltest(residuals)\n",
            "    print(f\"\\n=== Teste de Normalidade (D'Agostino-Pearson) para {model_name} ===\")\n",
            "    print(f\"Estatística de teste: {stat:.4f}\")\n",
            "    print(f\"p-valor: {p_val:.4e}\")\n",
            "    if p_val < 0.05:\n",
            "        print(\"Rejeita-se a hipótese nula H0 de que os resíduos são normalmente distribuídos (p < 0.05).\")\n",
            "    else:\n",
            "        print(\"Não se rejeita H0. Os resíduos seguem uma distribuição aproximadamente normal.\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.2 Análise de Resíduos vs. Outras Features (`hr` e `season`)\n",
            "\n",
            "Além da temperatura, vamos avaliar o comportamento dos resíduos em relação às features cruciais `hr` (hora do dia) e `season` (estação), permitindo diagnosticar se o modelo tem um desempenho inferior em faixas de horários específicos."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for model_name in top_2_names:\n",
            "    pipeline = best_estimators[model_name]\n",
            "    y_pred_log = pipeline.predict(X_test)\n",
            "    y_pred = np.expm1(y_pred_log)\n",
            "    residuals = y_test - y_pred\n",
            "    \n",
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    fig.suptitle(f\"Resíduos vs. Features Temporais: {model_name}\", fontsize=15)\n",
            "    \n",
            "    # Resíduos vs Hora do Dia\n",
            "    sns.boxplot(x=X_test['hr'], y=residuals, ax=axes[0], color='lightblue')\n",
            "    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)\n",
            "    axes[0].set_title(\"Resíduos por Hora do Dia\")\n",
            "    axes[0].set_xlabel(\"Hora do Dia\")\n",
            "    axes[0].set_ylabel(\"Resíduo\")\n",
            "    \n",
            "    # Resíduos vs Estação\n",
            "    sns.boxplot(x=X_test['season'], y=residuals, ax=axes[1], color='lightgreen')\n",
            "    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)\n",
            "    axes[1].set_title(\"Resíduos por Estação\")\n",
            "    axes[1].set_xlabel(\"Estação\")\n",
            "    axes[1].set_ylabel(\"Resíduo\")\n",
            "    axes[1].set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'])\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    filename = f\"residuos_temporal_{model_name.lower().replace(' ', '_')}.png\"\n",
            "    fig.savefig(f\"../relatorio/img/{filename}\", dpi=150, bbox_inches=\"tight\")\n",
            "    plt.show()"
        ]
    })
    
    # Write to file
    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)
    notebooks_dir = os.path.join(workspace_root, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    notebook_path = os.path.join(notebooks_dir, "experimentos.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1, ensure_ascii=False)
    print(f"Notebook gerado em {notebook_path}")

if __name__ == "__main__":
    build_notebook()
