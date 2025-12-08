#%%
# Importando bibliotecas:

import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff

warnings.simplefilter('ignore')

#%%
# Importação dos dados:

f_zip= 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'

r= requests.get(f_zip, stream=True)

vertebral_zip= zipfile.ZipFile(io.BytesIO(r.content))

vertebral_zip.extractall()

#%%
# Leitura de arquivos:

data= arff.loadarff('C:\Michel\Curso\dataScience\column_2C_weka.arff')

df= pd.DataFrame(data[0])

#%%
# Exploração de dados:
print(df.shape)

#%%
# Listando colunas:
print(df.columns)

#%%
# Verificando tipos de coluna:
print(df.dtypes)

#%%
# Examinando estatísticas da primeira coluna:
print(df['pelvic_incidence'].describe())

#%%
# Exibindo a estatística de cada componente:
print(df.describe)

#%%
# Plotando os valores:
import matplotlib.pyplot as plt

df.plot()
plt.show()

#%%
# Plotando valores para cada componente:
df.plot(kind='density', subplots=True, layout=(4,2), figsize=(12,12), sharex=False)
plt.show()

#%%
# Investiganndo a visualização:
df['degree_spondylolisthesis'].plot.density()
plt.show()

#%%
# Visualizando dados com um histograma
df['degree_spondylolisthesis'].plot.hist()
plt.show()

#%%
# Visualização de anomalias:
df['degree_spondylolisthesis'].plot.box()
plt.show()

#%%
# Analisando target:
print(df['class'].value_counts())

#%%
# Mapeando:
class_mapper= {b'Abnormal':1, b'Normal':0}
df['class']= df['class'].replace(class_mapper)

#%%
# Plotando o degree_spondylolisthesis para o alvo (target)
df.plot.scatter(y='degree_spondylolisthesis', x='class')
plt.show()

#%%
# Visualização de multiplas variáveis:
df.groupby('class').boxplot(fontsize=20, rot=90, figsize=(20,10), patch_artist=True)
plt.show()

#%%
# Criando a matriz
corr_matrix= df.corr(numeric_only=True)
print(corr_matrix["class"].sort_values(ascending=False))

#%%
# Plotando os dados da matriz:
pd.plotting.scatter_matrix(df, figsize=(12,12))
plt.show()


#%%
# Visualizando correlações com o heatmap
import seaborn as sns

fig, ax= plt.subplots(figsize=(10,10))
colormap= sns.color_palette("BrBG", 10)
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")
plt.show()