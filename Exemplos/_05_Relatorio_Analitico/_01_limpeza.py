import pandas as pd
import unicodedata      # Usada para remover acentuação de textos

#%%
# Criação de um DataFrame com valores inconsistentes
# Observe que há variações em letras maiúsculas/minúsculas, acentuação e espaços
 
dados = pd.DataFrame({
	'turno': ['Manhã', 'Tarde', 'Noite', 'noite', 'NOITE', 'TARDE', 'manha'],
	'falha': ['sensor', 'Sensor', 'Erro humano', 'erro humano', 'Desgaste mecânico', 'sensor', 'desgaste Mecânico']
})
		
#%%
# Funções de tratamento de texto
		
# Função que remove acentos de um texto (ex: 'mecânico' vira 'mecanico')
def remover_acentos(txt):
	texto_normalizado = unicodedata.normalize('NFKD', txt)  # Decomposição dos caracteres acentuados
	texto_sem_acentos = texto_normalizado.encode('ASCII', 'ignore').decode('utf-8')  # Remove os acentos
	return texto_sem_acentos
		
# Função que padroniza os valores da coluna 'turno'
def padronizar_turno(valor):
	valor = valor.strip()                # Remove espaços em branco no início e fim
	valor = valor.lower()                # Converte todo o texto para minúsculas
	valor = remover_acentos(valor)       # Remove acentos
	valor = valor.capitalize()           # Coloca a primeira letra em maiúscula (ex: 'noite' → 'Noite')
	return valor
		
# Função que padroniza os valores da coluna 'falha'
def padronizar_falha(valor):
	valor = valor.strip()                # Remove espaços em branco
	valor = valor.lower()                # Converte todo o texto para minúsculas
	valor = remover_acentos(valor)       # Remove acentos
	return valor                         # Mantém tudo em minúsculas e sem acento
		
#%%
# Aplicação das funções de padronização
		
# Aplica a função de padronização para a coluna 'turno'
dados['turno'] = dados['turno'].apply(padronizar_turno)
		
# Aplica a função de padronização para a coluna 'falha'
dados['falha'] = dados['falha'].apply(padronizar_falha)
		
# Exibe as primeiras linhas do DataFrame para verificar se a padronização funcionou
dados.head()