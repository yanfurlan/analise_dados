# Projeto de Análise de Dados de Vendas

Este projeto é uma análise de dados fictícios de vendas de uma loja de varejo. O objetivo é coletar, limpar, explorar e modelar os dados para obter insights valiosos e fazer previsões de vendas.

# Estrutura do Projeto

Coleta de Dados: Carregar dados fictícios de um arquivo CSV.

Limpeza de Dados: Tratar valores ausentes, remover duplicatas e formatar colunas.

Análise Exploratória de Dados (EDA): Visualizar e analisar os dados.

Modelagem: Construir um modelo de previsão de vendas usando regressão linear.

Conclusões e Insights: Resumir as principais descobertas.

# Pré-requisitos

Python 3.6 ou superior

Bibliotecas Python: pandas, matplotlib, seaborn, scikit-learn

# Instalação

1. Clone este repositório:

git clone https://github.com/yanfurlan/analise_dados.git

2. Navegue até o diretório do projeto:

cd analise-dados

3. Instale as dependências:

pip install pandas matplotlib seaborn scikit-learn

# Uso

Execute o script principal para realizar a análise:

python analise_vendas.py

# Detalhes do Script

1. Coleta de Dados

Os dados são carregados a partir de um arquivo CSV hospedado online. Utilizamos o pandas para ler o arquivo:

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
data = pd.read_csv(url, sep='\t')

2. Limpeza de Dados

Verificamos valores ausentes, removemos duplicatas e formatamos colunas:

# Verificar valores ausentes
print(data.isnull().sum())

# Remover duplicatas
data = data.drop_duplicates()

# Converter preços para float
data['item_price'] = data['item_price'].str.replace('$', '').astype(float)

3. Análise Exploratória de Dados (EDA)

Realizamos análises estatísticas e visualizações:

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

4. Modelagem

Construímos um modelo de regressão linear para prever vendas:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

5. Conclusões e Insights

Resumimos os principais insights:

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

# Contribuição

Sinta-se à vontade para abrir issues e pull requests. Contribuições são bem-vindas!

# Licença

Este projeto está licenciado sob a Licença MIT.