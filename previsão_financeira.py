import pandas as pd
import numpy as np
from random import randint, uniform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Definir número de meses
np.random.seed(42)
meses = pd.date_range(start="2020-01-01", periods=48, freq='ME')

# Gerar dados simulados
base_dados = pd.DataFrame({
    'mes': meses,
    'numero_clientes': np.random.randint(500, 2000, len(meses)),
    'ticket_medio': np.random.uniform(50, 300, len(meses)),
    'custos_fixos': np.random.uniform(20000, 80000, len(meses)),
    'custos_variaveis': np.random.uniform(0.3, 0.7, len(meses)),  # Percentual sobre o faturamento
    'despesas_operacionais': np.random.uniform(5000, 30000, len(meses)),
    'investimentos': np.random.uniform(2000, 20000, len(meses))
})

# Calcular faturamento e lucro
base_dados['faturamento'] = base_dados['numero_clientes'] * base_dados['ticket_medio']
base_dados['lucro'] = base_dados['faturamento'] * (1 - base_dados['custos_variaveis']) - base_dados['custos_fixos'] - base_dados['despesas_operacionais']

# Salvar dataset
base_dados.to_csv("dados_financeiros.csv", index=False)

# Exibir as primeiras linhas
print(base_dados.head())

# Preparação dos dados para Machine Learning
X = base_dados[['numero_clientes', 'ticket_medio', 'custos_fixos', 'custos_variaveis', 'despesas_operacionais', 'investimentos']]
y_faturamento = base_dados['faturamento']
y_lucro = base_dados['lucro']

# Dividir os dados em treino e teste
X_train, X_test, y_fat_train, y_fat_test, y_lucro_train, y_lucro_test = train_test_split(
    X, y_faturamento, y_lucro, test_size=0.2, random_state=42
)

# Criar e treinar os modelos
modelo_faturamento = LinearRegression()
modelo_lucro = LinearRegression()
modelo_faturamento.fit(X_train, y_fat_train)
modelo_lucro.fit(X_train, y_lucro_train)

# Fazer previsões
fat_pred = modelo_faturamento.predict(X_test)
lucro_pred = modelo_lucro.predict(X_test)

# Avaliar os modelos
print("Faturamento - MAE:", mean_absolute_error(y_fat_test, fat_pred))
print("Faturamento - R²:", r2_score(y_fat_test, fat_pred))
print("Lucro - MAE:", mean_absolute_error(y_lucro_test, lucro_pred))
print("Lucro - R²:", r2_score(y_lucro_test, lucro_pred))

# Previsão de lucro anual
lucro_anual = base_dados['lucro'].sum()
print("Previsão de Lucro Anual:", lucro_anual)
