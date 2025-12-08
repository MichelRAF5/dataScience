import pandas as pd

#%%
# Lendo um arquivo CSV com separador ponto e vírgula:

df_csv = pd.read_csv("dados_temperatura.csv", sep=";", encoding="utf-8")
# sep= Define o separador usado (por exemplo vírgula "," ou ponto e vírgula ";")
# encoding= Define a codificação do arquivo (ex: "utf-8", "latin1")

# Exibindo as primeiras linhas
df_csv.head()

#%%
# Lendo uma planilha Excel, aba "abril" e colunas A a C:

df_excel = pd.read_excel("paradas_maquinas.xlsx", sheet_name="abril", usecols="A:C")
# sheet_name define a aba a ser lida
# usecols permite selecionar apenas as colunas de interesse

# Exibindo as primeiras linhas
df_excel.head()

#%%
# Leitura e normalização de dados aninhados em arquivos json:
import json

# Abrindo o arquivo
with open("arquivo.json") as f:
    dados = json.load(f)
	 
# Normalizando os dados aninhados
df = pd.json_normalize(dados, record_path=["registros"], meta=["setor"])
	 
# Exibindo as primeiras linhas
df.head()