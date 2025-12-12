#%%
import pandas as pd

# Define as opções de exibição do pandas
pd.set_option('display.max_rows', 500)     # Define o número de linhas que serão exibidas
pd.set_option('display.max_columns', 500)  # Define o número de colunas que serão exibidas
pd.set_option('display.width', 1000)       # Define a largura máxima de exibição dos caracteres

import warnings , requests, zipfile, io
from scipy.io import arff

#%%
# Carregamento dos dados

url= "C:\Michel\Curso\dataScience\imports-85.data"

col_names= ['symboling', 'normalized-losses', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base',
            'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 
            'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# Leitura do arquivo csv para criar o Data Frame df_car
df_car= pd.read_csv(url, sep=',', names= col_names, na_values='?', header=None)

#%%
# Verificando o shape
print(df_car.shape)

#%%
# Examinando os dados
print(df_car.head)

#%%
# Exibindo as innformações
df_car.info()

#%%
# Exibindo as colunas
print(df_car.columns)

#%%
# Copiando colunas
df_car= df_car[['aspiration', 'num-of-doors', 'drive-wheels', 'num-of-cylinders']].copy()

#%%
# Examinando os dados
print(df_car.head)

#%%
# Codificando as informações
df_car.info()

#%%
# Determinando valores ordinais
print(df_car['num-of-doors'].value_counts)

#%%
# Definindo um dicionário (mapper) para converter representações
door_mapper= {"two": 2,
              "four": 4}

#%%
# Gerando a nova coluna
# Aplicando o dicionário de mapeamento 'door_mapper' à coluna estratégica
df_car['doors']= df_car['num-of-doors'].replace(door_mapper)
print(df_car['doors'])

#%%
# Exibindo o novo  dataframe
print(df_car.head())

#%%
# Gerando novas colunas
print(df_car['num-of-cylinders'].value_counts())

# Criando o mapeador
cylinder_mapper= {"two": 2,
                  "three": 3,
                  "four": 4,
                  "five": 5,
                  "six": 6,
                  "eight": 8,
                  "twelve": 12}

df_car['cylinders']= df_car['num-of-cylinders'].replace(cylinder_mapper)


#%%
# Exibindo o DataFrame
print(df_car)

#%%
# Codificando os dados não ordinais
print(df_car['drive-wheels'].value_counts())

#%%
# Adicionando novos componentes
df_car= pd.get_dummies(df_car, columns=['drive-wheels'])    # Converte as variáveis categóricas em colunas das categorias com valores booleanos
print(df_car.head())

#%%
# Calculando a contagem de frequência
print(df_car['aspiration'].value_counts())

#%%
# Aplicando o One-Hot encoder
df_car= pd.get_dummies(
                        df_car,                      # Data frame de entrada
                        columns= ['aspiration'],     # Coluna categórica a ser codificada
                        drop_first= True             # Parâmetro que determina se 'aspiration' tem os valores 'std' e 'turbo', esta opção cria apenas uma coluna binária (aspiration_turbo que será True ou False) 
                    )


#%%
# Visualização do One-Hot encoder
print(df_car.head())