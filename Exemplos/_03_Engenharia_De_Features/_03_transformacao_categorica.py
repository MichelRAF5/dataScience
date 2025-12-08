#%%
# As variáveis categóricas representam valores não numéricos, como nomes de turnos, tipos de equipamento ou status de operação.
# Para que esses dados possam ser usados em modelos, é necessário transformá-los em números, mas com cuidado.
# Modelos como regressão linear ou redes neurais não interpretam texto diretamente. 
# Por isso, duas técnicas são usadas para converter essas categorias em um formato adequado para análise:

# Label Encoding:
# Simples e transforma texto em número. Porém, pode induzir uma ordem em um ponto onde ela não existe.

# One-Hot Encoding:
# Cria uma coluna para cada categoria, usando valores 0 ou 1. É a opção mais segura quando não há hierarquia entre os valores, mas aumenta a dimensionalidade.

#%%
import pandas as pd
	 
# Criando um DataFrame com uma coluna categórica de exemplo
df = pd.DataFrame({
	'maquina': ['M1', 'M2', 'M3', 'M4'],
	'turno': ['manha', 'noite', 'manha', 'tarde']
})
	 
print("DataFrame original:")
print(df)
	 
# Label Encoding: converte categorias para números inteiros
df['turno_label'] = df['turno'].astype('category').cat.codes
print("\nCom Label Encoding (turno_label):")
print(df)
	 
# One-Hot Encoding: cria colunas binárias para cada categoria
df_onehot = pd.get_dummies(df['turno'], prefix='turno')
print("\nCom One-Hot Encoding:")
print(df_onehot)
	 
# Opcional: unindo ao DataFrame original para comparação
df_completo = pd.concat([df, df_onehot], axis=1)
print("\nDataFrame completo com as codificações:")
print(df_completo)

#%%
# A escolha entre uma técnica e outra depende do tipo de modelo.
# Árvores de decisão, por exemplo, lidam bem com labels numéricos.
# Já redes neurais e regressões funcionam melhor com codificação binária.
