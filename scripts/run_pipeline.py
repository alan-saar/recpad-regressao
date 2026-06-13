import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import joblib

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run():
    print("=== Iniciando execução do Pipeline ===")
    
    # 1. Carregar dados
    print("Buscando dados da UCI...")
    dataset = fetch_ucirepo(id=275)
    X = dataset.data.features.drop(columns=['dteday'])
    y = dataset.data.targets['cnt']
    
    df = X.copy()
    df['cnt'] = y
    
    # Criar pastas se não existirem
    os.makedirs("relatorio/img", exist_ok=True)
    os.makedirs("modelos_salvos", exist_ok=True)
    
    # ----------------------------------------------------
    # Etapa 1: EDA e Salvar Gráficos
    # ----------------------------------------------------
    print("Executando EDA e salvando gráficos...")
    
    # Gráfico 1: Distribuição de cnt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['cnt'], bins=50, kde=True, color='steelblue', edgecolor='white', ax=axes[0])
    axes[0].set_title('Distribuição de cnt (Original)')
    axes[0].set_xlabel('cnt (Aluguéis de Bicicletas)')
    axes[0].set_ylabel('Frequência')
    
    sns.histplot(np.log1p(df['cnt']), bins=50, kde=True, color='darkorange', edgecolor='white', ax=axes[1])
    axes[1].set_title('Distribuição de log1p(cnt)')
    axes[1].set_xlabel('log1p(cnt)')
    axes[1].set_ylabel('Frequência')
    plt.tight_layout()
    fig.savefig("relatorio/img/distribuicao_cnt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Gráfico 2: Heatmap de Correlações
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title('Mapa de Correlações')
    plt.tight_layout()
    fig.savefig("relatorio/img/heatmap_correlacoes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Gráfico 3: Padrões Temporais
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df.groupby('hr')['cnt'].mean().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Média de Aluguéis por Hora')
    axes[0].set_xlabel('Hora do Dia')
    axes[0].set_ylabel('cnt Médio')
    axes[0].tick_params(axis='x', rotation=0)
    
    df.groupby('mnth')['cnt'].mean().plot(kind='bar', ax=axes[1], color='seagreen', edgecolor='white')
    axes[1].set_title('Média de Aluguéis por Mês')
    axes[1].set_xlabel('Mês')
    axes[1].set_ylabel('cnt Médio')
    axes[1].tick_params(axis='x', rotation=0)
    
    df.groupby('season')['cnt'].mean().plot(kind='bar', ax=axes[2], color='darkorange', edgecolor='white')
    axes[2].set_title('Média de Aluguéis por Estação')
    axes[2].set_xlabel('Estação')
    axes[2].set_ylabel('cnt Médio')
    axes[2].tick_params(axis='x', rotation=0)
    axes[2].set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'])
    plt.tight_layout()
    fig.savefig("relatorio/img/padroes_temporais.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Gráfico 4: Boxplots Clima e Contexto
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.boxplot(data=df, x='weathersit', y='cnt', ax=axes[0], palette='Blues')
    axes[0].set_title('cnt por Condição Climática')
    axes[0].set_xlabel('weathersit (1=Claro → 4=Chuva Forte)')
    axes[0].set_ylabel('cnt')
    
    sns.boxplot(data=df, x='workingday', y='cnt', ax=axes[1], palette='Set1')
    axes[1].set_title('cnt — Dia Útil vs. Não Útil')
    axes[1].set_xticklabels(['Fim de Semana/Feriado', 'Dia Útil'])
    axes[1].set_xlabel('Tipo de Dia')
    axes[1].set_ylabel('cnt')
    
    sns.boxplot(data=df, x='weekday', y='cnt', ax=axes[2], palette='Set2')
    axes[2].set_title('cnt por Dia da Semana')
    axes[2].set_xlabel('Dia (0=Domingo → 6=Sábado)')
    axes[2].set_ylabel('cnt')
    plt.tight_layout()
    fig.savefig("relatorio/img/clima_e_contexto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ----------------------------------------------------
    # Etapa 2: Pré-processamento
    # ----------------------------------------------------
    print("Executando Pré-processamento...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X['season']
    )
    
    num_cols = ['temp', 'atemp', 'hum', 'windspeed']
    cat_ord_cols = ['hr', 'mnth', 'season', 'weekday', 'weathersit']
    bin_cols = ['holiday', 'workingday', 'yr']
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('ord', 'passthrough', cat_ord_cols),
            ('bin', 'passthrough', bin_cols)
        ]
    )
    
    # ----------------------------------------------------
    # Etapa 3: Ajuste de Modelos (Tuning com 5-fold CV)
    # ----------------------------------------------------
    print("Ajustando modelos (RandomizedSearchCV)...")
    cv_tuning = KFold(n_splits=5, shuffle=True, random_state=42)
    best_estimators = {}
    
    # 1. OLS
    print("- OLS...")
    ols_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    ols_pipeline.fit(X_train, y_train_log)
    best_estimators['OLS'] = ols_pipeline
    
    # 2. Ridge
    print("- Ridge...")
    ridge_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])
    ridge_search = RandomizedSearchCV(
        ridge_pipeline, param_distributions={'regressor__alpha': stats.loguniform(1e-3, 1e3)},
        n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    ridge_search.fit(X_train, y_train_log)
    best_estimators['Ridge'] = ridge_search.best_estimator_
    print(f"  Ridge melhor alpha: {ridge_search.best_params_['regressor__alpha']:.4f}")
    
    # 3. Lasso
    print("- Lasso...")
    lasso_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(max_iter=10000))
    ])
    lasso_search = RandomizedSearchCV(
        lasso_pipeline, param_distributions={'regressor__alpha': stats.loguniform(1e-5, 1.0)},
        n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    lasso_search.fit(X_train, y_train_log)
    best_estimators['Lasso'] = lasso_search.best_estimator_
    print(f"  Lasso melhor alpha: {lasso_search.best_params_['regressor__alpha']:.6f}")
    
    # 4. Random Forest
    print("- Random Forest (pode demorar)...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    rf_param_dist = {
        'regressor__n_estimators': stats.randint(50, 150),
        'regressor__max_depth': stats.randint(8, 20),
        'regressor__min_samples_split': stats.randint(2, 10),
        'regressor__max_features': ['sqrt', None]
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline, param_distributions=rf_param_dist,
        n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train_log)
    best_estimators['Random Forest'] = rf_search.best_estimator_
    print(f"  Random Forest melhores parâmetros: {rf_search.best_params_}")
    
    # 5. XGBoost
    print("- XGBoost...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    xgb_param_dist = {
        'regressor__n_estimators': stats.randint(80, 200),
        'regressor__max_depth': stats.randint(4, 8),
        'regressor__learning_rate': stats.uniform(0.01, 0.20),
        'regressor__subsample': stats.uniform(0.6, 0.4),
        'regressor__colsample_bytree': stats.uniform(0.6, 0.4),
        'regressor__reg_alpha': stats.loguniform(1e-3, 10.0),
        'regressor__reg_lambda': stats.loguniform(1e-3, 10.0)
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipeline, param_distributions=xgb_param_dist,
        n_iter=30, cv=cv_tuning, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train_log)
    best_estimators['XGBoost'] = xgb_search.best_estimator_
    print(f"  XGBoost melhores parâmetros: {xgb_search.best_params_}")
    
    # Salvar modelos
    for name, est in best_estimators.items():
        joblib.dump(est, f"modelos_salvos/{name.lower().replace(' ', '_')}_model.pkl")
    
    # ----------------------------------------------------
    # Etapa 4: Avaliação Experimental (10-fold CV & Hold-out)
    # ----------------------------------------------------
    print("Avaliando modelos...")
    
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100
        return rmse, mae, r2, mape

    cv_results = []
    holdout_results = []
    cv_10fold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for name, pipeline in best_estimators.items():
        # CV 10-folds manual
        rmse_folds, mae_folds, r2_folds, mape_folds = [], [], [], []
        for train_idx, val_idx in cv_10fold.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_log, y_val = y_train_log.iloc[train_idx], y_train.iloc[val_idx]
            
            from sklearn.base import clone
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_tr, y_tr_log)
            
            y_val_pred = np.expm1(fold_pipeline.predict(X_val))
            rmse_f, mae_f, r2_f, mape_f = calculate_metrics(y_val, y_val_pred)
            rmse_folds.append(rmse_f)
            mae_folds.append(mae_f)
            r2_folds.append(r2_f)
            mape_folds.append(mape_f)
            
        cv_results.append({
            'Modelo': name,
            'RMSE CV': f"{np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}",
            'MAE CV': f"{np.mean(mae_folds):.2f} ± {np.std(mae_folds):.2f}",
            'R2 CV': f"{np.mean(r2_folds):.3f} ± {np.std(r2_folds):.3f}",
            'MAPE CV (%)': f"{np.mean(mape_folds):.2f}% ± {np.std(mape_folds):.2f}%"
        })
        
        # Teste Hold-out
        y_test_pred = np.expm1(pipeline.predict(X_test))
        rmse_t, mae_t, r2_t, mape_t = calculate_metrics(y_test, y_test_pred)
        holdout_results.append({
            'Modelo': name,
            'RMSE Teste': rmse_t,
            'MAE Teste': mae_t,
            'R2 Teste': r2_t,
            'MAPE Teste (%)': mape_t
        })
        
    df_cv = pd.DataFrame(cv_results)
    df_holdout = pd.DataFrame(holdout_results).sort_values(by='RMSE Teste')
    
    print("\n=== Resultados 10-Fold CV ===")
    print(df_cv.to_string(index=False))
    print("\n=== Resultados Hold-out no Teste ===")
    print(df_holdout.to_string(index=False))
    
    df_cv.to_csv("relatorio/img/resultado_cv.csv", index=False)
    df_holdout.to_csv("relatorio/img/resultado_holdout.csv", index=False) # Note: CSV save
    # Let's fix that string
    
    # ----------------------------------------------------
    # Etapa 5: Diagnóstico dos Resíduos (2 melhores)
    # ----------------------------------------------------
    top_2_names = df_holdout['Modelo'].head(2).tolist()
    print(f"\nFazendo diagnóstico de resíduos para os dois melhores: {top_2_names}")
    
    for model_name in top_2_names:
        pipeline = best_estimators[model_name]
        y_pred = np.expm1(pipeline.predict(X_test))
        residuals = y_test - y_pred
        
        # Gráficos de Resíduos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Análise de Resíduos: {model_name}", fontsize=16)
        
        # Histograma
        sns.histplot(residuals, kde=True, bins=50, color='darkred', edgecolor='white', ax=axes[0, 0])
        axes[0, 0].set_title("Distribuição dos Resíduos")
        axes[0, 0].set_xlabel("Resíduo (Real - Predito)")
        
        # Scatter Preditos
        axes[0, 1].scatter(y_pred, residuals, alpha=0.3, color='purple', edgecolors='none')
        axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
        axes[0, 1].set_title("Resíduos vs. Valores Preditos")
        axes[0, 1].set_xlabel("Valores Preditos")
        axes[0, 1].set_ylabel("Resíduo")
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        
        # Temp Scatter
        axes[1, 1].scatter(X_test['temp'], residuals, alpha=0.3, color='teal', edgecolors='none')
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
        axes[1, 1].set_title("Resíduos vs. Temperatura")
        axes[1, 1].set_xlabel("Temperatura (Normalizada)")
        axes[1, 1].set_ylabel("Resíduo")
        
        plt.tight_layout()
        m_id = model_name.lower().replace(' ', '_')
        fig.savefig(f"relatorio/img/residuos_{m_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Gráficos adicionais temporais
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Resíduos vs. Features Temporais: {model_name}", fontsize=15)
        
        sns.boxplot(x=X_test['hr'], y=residuals, ax=axes[0], color='lightblue')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
        axes[0].set_title("Resíduos por Hora do Dia")
        axes[0].set_xlabel("Hora")
        axes[0].set_ylabel("Resíduo")
        
        sns.boxplot(x=X_test['season'], y=residuals, ax=axes[1], color='lightgreen')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
        axes[1].set_title("Resíduos por Estação")
        axes[1].set_xlabel("Estação")
        axes[1].set_ylabel("Resíduo")
        axes[1].set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'])
        
        plt.tight_layout()
        fig.savefig(f"relatorio/img/residuos_temporal_{m_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Teste de Normalidade
        stat, p_val = stats.normaltest(residuals)
        print(f"Normality Test for {model_name}: Stat={stat:.4f}, p-value={p_val:.4e}")
        
    print("=== Pipeline executado com sucesso e todos os artefatos salvos! ===")

if __name__ == "__main__":
    run()
