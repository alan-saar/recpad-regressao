#import "modelo/IFES-DOC/ifes-doc.typ": *

#show: setup.with(
  title: [Reconhecimento de Padrões - Trabalho 2],
  subtitle: [Análise Comparativa de Métodos de Regressão],
  author: "Alan P. Berger Saar e Leonardo Herkenhoff",
  affiliation: "PPCOMP - IFES",
  course: "Reconhecimento de Padrões",
  date: "2026",
)

= Introdução

O presente relatório descreve o desenvolvimento e a análise comparativa de métodos de regressão aplicados à base de dados *Bike Sharing* (UCI \#275, versão _hourly_). Os sistemas de aluguel de bicicletas desempenham um papel relevante na mobilidade urbana moderna, gerando grandes volumes de dados que capturam padrões de fluxo de tráfego de acordo com condições climáticas, sazonais e de calendário.

Prever com precisão a demanda horária de aluguel de bicicletas (representada pela variável-alvo `cnt`) permite que operadores planejem a realocação de frotas e dimensionem serviços de forma mais eficiente. Este trabalho aborda o problema sob a perspectiva de regressão tabular, avaliando cinco modelos de regressão de complexidades crescentes:
1. *Mínimos Quadrados Ordinários (OLS)* — baseline linear clássico.
2. *Regressão Ridge* — modelo linear com regularização L2.
3. *Regressão Lasso* — modelo linear com regularização L1 (seleção de features).
4. *Random Forest Regressor* — modelo não linear de ensemble baseado em bagging.
5. *XGBoost Regressor* — modelo não linear de ensemble baseado em gradient boosting.

O relatório está estruturado da seguinte forma: a Seção 2 apresenta a análise exploratória dos dados; a Seção 3 detalha a metodologia, incluindo o pré-processamento e protocolo de tuning; a Seção 4 traz os resultados experimentais e a análise dos resíduos; a Seção 5 desenvolve uma discussão crítica; e a Seção 6 conclui o estudo apresentando as limitações identificadas.

= Dataset e Análise Exploratória dos Dados (EDA)

A base de dados *Bike Sharing* contém $17.379$ registros horários compreendidos entre os anos de 2011 e 2012. Cada registro é composto por 12 características (excluindo a coluna `dteday` que foi descartada por não ser numérica). As características incluem dados temporais (ano, mês, estação, hora e dia da semana) e meteorológicos (temperatura real, sensação térmica, umidade e velocidade do vento).

== Distribuição da Variável-Alvo e Transformação Logarítmica

A distribuição de frequência da variável `cnt` (contagem total de aluguéis) é fortemente assimétrica à direita (skewness original de $1.28$). Em problemas de regressão, distribuições com cauda longa violam as hipóteses de homocedasticidade e normalidade de resíduos para modelos lineares, fazendo com que o erro do modelo aumente proporcionalmente à magnitude do valor real.

Para contornar este problema, aplicamos a transformação $y' = log(1 + y)$ (conhecida como `log1p`), que estabiliza a variância e aproxima a distribuição de uma normal, reduzindo o skewness para $-1.10$. A reversão é feita com $y = exp(y') - 1$ (`expm1`) para relatar as métricas em bicicletas/hora.

#figure(
  image("img/distribuicao_cnt.png", width: 90%),
  caption: [Distribuição de freqüência de cnt antes e após a transformação log1p.],
) <fig-dist-cnt>

== Correlações e Multicolinearidade

O mapa de calor de correlações de Pearson (@fig-heatmap) revela que as variáveis `temp` (temperatura real normalizada) e `atemp` (sensação térmica normalizada) possuem uma correlação linear quase perfeita ($r = 0.99$). Esta forte colinearidade cria instabilidade em modelos de regressão linear clássicos (OLS), elevando a variância dos coeficientes estimadores. A inclusão de termos de regularização L1/L2 visa estabilizar essas estimativas.

#figure(
  image("img/heatmap_correlacoes.png", width: 70%),
  caption: [Mapa de calor com coeficientes de correlação de Pearson.],
) <fig-heatmap>

== Padrões Temporais e Climáticos

A demanda por bicicletas exibe perfis altamente sazonais. Há um consumo médio significativamente maior nos meses centrais do ano (verão/outono), influenciado por temperaturas favoráveis, e uma forte variação horária com picos nítidos em dias de semana nos horários de entrada e saída laboral/escolar (~8h e ~17h–18h). Condições climáticas desfavoráveis (`weathersit` maior ou igual a 3) reduzem dramaticamente o número de aluguéis.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("img/padroes_temporais.png", width: 100%),
    caption: [Consumo médio por hora, mês e estação.],
  ),
  figure(
    image("img/clima_e_contexto.png", width: 100%),
    caption: [Boxplots por clima, dia de semana e dia útil.],
  )
)

= Metodologia

O pipeline experimental foi estruturado de forma reprodutível e livre de vazamento de dados (*data leakage*):

1. *Divisão Hold-out*: Separação de 80% dos dados para treino ($13.903$ instâncias) e 20% para teste ($3.476$ instâncias), estratificados pela variável `season` para manter as proporções sazonais idênticas em ambos os conjuntos.
2. *Pré-processamento*:
   - Padronização (`StandardScaler`) aplicada somente nas features numéricas contínuas (`temp`, `atemp`, `hum`, `windspeed`). Os parâmetros estatísticos ($mu$ e $sigma$) foram estimados unicamente no treino e aplicados no teste.
   - Variáveis categóricas ordinais (`hr`, `mnth`, `season`, `weekday`, `weathersit`) foram mantidas em escala ordinal direta.
   - Variáveis binárias (`holiday`, `workingday`, `yr`) foram passadas inalteradas.
3. *Modelagem e Tuning*:
   - Realizou-se busca hiperparamétrica usando `RandomizedSearchCV` com 5-fold CV no conjunto de treino ($n_("iter") = 30$).
   - Os melhores hiperparâmetros encontrados foram:
     - *Ridge*: $alpha = 24.6583$
     - *Lasso*: $alpha = 0.000679$
     - *Random Forest*: `max_depth = 17`, `max_features = None`, `min_samples_split = 3`, `n_estimators = 123`
     - *XGBoost*: `n_estimators = 165`, `max_depth = 6`, `learning_rate = 0.1298`, `subsample = 0.7183`, `colsample_bytree = 0.9771`, `reg_alpha = 3.3256`, `reg_lambda = 0.3144`

= Resultados

A avaliação experimental comparou os modelos sob duas óticas: estabilidade na validação cruzada de 10 folds e desempenho no conjunto de teste Hold-out. Todas as métricas foram calculadas no espaço original da variável-alvo (após aplicação de `expm1`).

#styled-table(
  headers: ("Modelo", "RMSE CV", "MAE CV", "R² CV", "MAPE CV (%)"),
  rows: (
    ("OLS", "162.78 ± 3.29", "108.22 ± 2.24", "0.197 ± 0.023", "148.03% ± 7.78%"),
    ("Ridge", "162.76 ± 3.30", "108.24 ± 2.24", "0.197 ± 0.023", "148.05% ± 7.77%"),
    ("Lasso", "162.74 ± 3.30", "108.21 ± 2.24", "0.198 ± 0.023", "148.07% ± 7.81%"),
    ("Random Forest", "44.20 ± 2.37", "26.13 ± 0.83", "0.941 ± 0.007", "27.17% ± 2.55%"),
    ("XGBoost", "41.60 ± 2.80", "25.06 ± 1.09", "0.947 ± 0.008", "24.32% ± 1.27%"),
  )
)

#styled-table(
  headers: ("Modelo (Teste)", "RMSE Teste", "MAE Teste", "R² Teste", "MAPE Teste (%)"),
  rows: (
    ("XGBoost", "41.08", "24.38", "0.948", "23.84%"),
    ("Random Forest", "41.92", "24.77", "0.946", "25.51%"),
    ("Ridge", "159.61", "106.10", "0.213", "144.57%"),
    ("OLS", "159.62", "106.09", "0.213", "144.58%"),
    ("Lasso", "159.64", "106.11", "0.213", "144.60%"),
  )
)

Os resultados deixam evidente que os modelos não lineares de ensemble (*XGBoost* e *Random Forest*) são ordens de grandeza superiores aos modelos lineares tradicionais. O XGBoost alcançou o melhor desempenho, com menor erro (RMSE = $41.08$) e capacidade de explicar $94.8\%$ da variância total da demanda no conjunto de teste.

== Diagnóstico dos Resíduos

O diagnóstico de resíduos no conjunto de teste foi conduzido para os dois melhores modelos (XGBoost e Random Forest). 

#figure(
  image("img/residuos_xgboost.png", width: 90%),
  caption: [Gráficos de resíduos para o XGBoost: histograma, dispersão vs. preditos, Q-Q Plot e dispersão vs. temperatura.],
) <fig-res-xgb>

#figure(
  image("img/residuos_random_forest.png", width: 90%),
  caption: [Gráficos de resíduos para o Random Forest.],
) <fig-res-rf>

=== Teste Formal de Normalidade

Aplicamos o teste de D'Agostino-Pearson (`stats.normaltest`) para verificar formalmente a hipótese nula $H_0$ de normalidade dos resíduos:
- *XGBoost*: Estatística de teste = $1092.56$, $p$-valor = $5.67 dot 10^(-238)$.
- *Random Forest*: Estatística de teste = $1224.65$, $p$-valor = $1.18 dot 10^(-266)$.

Ambos os modelos rejeitam fortemente a hipótese nula de normalidade dos resíduos ($p < 0.05$). O desvio em relação à normalidade teórica ocorre principalmente nas caudas pesadas evidentes nos gráficos Q-Q Plot.

#figure(
  image("img/residuos_temporal_xgboost.png", width: 90%),
  caption: [Distribuição de resíduos por hora e estação para o XGBoost.],
) <fig-res-temp-xgb>

= Discussão

A enorme discrepância no desempenho de modelos lineares (OLS, Ridge e Lasso com R² $approx 0.21$ no teste) em comparação com ensembles não lineares (R² $approx 0.95$) revela a natureza fortemente não linear do dataset. Padrões diários de pico e a relação de temperatura em formato de sino não podem ser capturados por uma combinação linear simples de suas características ordinais/contínuas. 

A regularização linear (Lasso vs. Ridge) não resultou em ganhos significativos porque o modelo linear clássico é essencialmente limitado por seu viés de representação neste dataset. O Lasso obteve $alpha$ extremamente pequeno ($0.000679$), não promovendo a eliminação efetiva de features por limitação de capacidade geral de ajuste.

O diagnóstico de resíduos revelou comportamentos interessantes:
1. *Heterocedasticidade*: O gráfico de resíduos vs. valores preditos mostra uma leve expansão da dispersão à medida que o volume de aluguéis aumenta. O erro de predição é maior para demandas altas.
2. *Padrão nos resíduos vs. features temporais* (@fig-res-temp-xgb): Os boxplots de resíduos por hora do dia exibem maior dispersão (maior variância do erro) especificamente nos horários de pico (8h e 17h-18h), período em que a variabilidade de comportamento do usuário também é mais alta.

O aumento drástico na complexidade computacional imposto pelo Random Forest e XGBoost é totalmente justificado pelo salto de desempenho (redução do RMSE de $approx 159$ para $approx 41$).

= Conclusão e Limitações

Este estudo demonstrou a aplicabilidade de técnicas avançadas de aprendizado de máquina para modelagem e previsão de demanda em sistemas de micromobilidade. Os ensembles baseados em árvores (Random Forest e XGBoost) provaram ser modelos altamente flexíveis e adequados ao problema, apresentando elevada qualidade de ajuste ($R^2 > 0.945$).

No entanto, o estudo possui limitações fundamentais:
- *Tratamento Tabular Simples*: Tratar o dataset cronológico como dados tabulares puramente estocásticos ignora o efeito de autocorrelação temporal das séries temporais. Em um cenário prático, a ordem dos dados (dia anterior, hora anterior) carrega forte poder informativo.
- *Data Leakage na Divisão*: A divisão hold-out aleatória permitiu que observações do mesmo dia caíssem simultaneamente no treino e no teste. Isso reduz a capacidade de prever a demanda em dias completamente futuros e pode superestimar levemente o desempenho relatado.

Para trabalhos futuros, recomenda-se a estruturação do problema sob o prisma de previsão de séries temporais com validação temporal sequencial (_Time Series Split_), além de testar encoding cíclico (seno e cosseno) para as variáveis temporais como `hr` e `weekday`.

= Referências

Fanaee-T, H. & Gama, J. (2014). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence, 2(2–3), 113–127. https://doi.org/10.1007/s13748-013-0040-3
