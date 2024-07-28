import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Coleta de Dados
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
data = pd.read_csv(url, sep='\t')

# 2. Limpeza de Dados
# Verificar valores ausentes
print(data.isnull().sum())

# Remover duplicatas
data = data.drop_duplicates()

# Converter colunas para o tipo adequado (exemplo: converter preços para float)
data['item_price'] = data['item_price'].str.replace('$', '').astype(float)

# 3. Análise Exploratória de Dados (EDA)
# Descrição estatística dos dados
print(data.describe())

# Visualização das vendas por item
plt.figure(figsize=(10, 6))
sns.countplot(y='item_name', data=data, order=data['item_name'].value_counts().index)
plt.title('Número de Vendas por Item')
plt.xlabel('Número de Vendas')
plt.ylabel('Item')
plt.show()

# Total de vendas por item
total_sales = data.groupby('item_name')['item_price'].sum().sort_values(ascending=False)
print(total_sales)

# 4. Modelagem
# Usar apenas colunas numéricas para o modelo
data['order_id'] = data['order_id'].astype(int)
X = data[['order_id', 'quantity']]
y = data['item_price']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# 5. Conclusões e Insights
# Principais insights
print("Principais Insights:")
print(f"Item mais vendido: {data['item_name'].value_counts().idxmax()}")
print(f"Item que gerou maior receita: {total_sales.idxmax()}")
print(f"Mean Squared Error do modelo: {mse}")

# Visualização das previsões vs. valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs. Valores Reais')
plt.show()
