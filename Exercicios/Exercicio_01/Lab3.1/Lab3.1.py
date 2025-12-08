#%%
# Importando bibliotecas:

import warnings, requests, zipfile, io
# warnings(avisos)
# requests (requisições web)
# zipfile (arquivos ZIP)
# io (entrada e saída em memória)

warnings.simplefilter('ignore')
# ignorar mensagens de avisos para não poluir o programa

import pandas as pd
from scipy.io import arff # Importa a função para importar arquivos ARFF

#%%
# Importação dos dados:

f_zip= 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'

r= requests.get(f_zip, stream=True)
# requests.get: faz o dowload do arquivo zip
# stream=True: baixa o coonteúde em partes (útil para arquivos grandes)

vertebral_zip= zipfile.ZipFile(io.BytesIO(r.content))
# Converte o  conteúdo baixado em um arquivo de memória
# e abre como um arquivo ZIP para manipulação

vertebral_zip.extractall()

#%%
# Lendo os arquivos:

data= arff.loadarff('C:\Michel\Curso\dataScience\column_2C_weka.arff')

df= pd.DataFrame(data[0])
print(df.head())