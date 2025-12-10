import matplotlib.pyplot as plt
import _01_limpeza
	 
# Contagem de falhas por turno
falhas_turno = _01_limpeza.dados['turno'].value_counts().sort_index()
	 
# Cores acessíveis: azul, laranja e verde - paleta colorblind-friendly (CBF)
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib default CBF palette
	 
# Plotagem do gráfico
plt.figure(figsize=(6, 4))
plt.bar(falhas_turno.index, falhas_turno.values, color=cores)
	 
# Título e eixos
plt.title("Ocorrência de falhas por turno")
plt.xlabel("Turno")
plt.ylabel("Número de falhas")
	 
# Grade para facilitar leitura
plt.grid(axis='y', linestyle='--', alpha=0.5)
	 
# Apresentação final
plt.tight_layout()
plt.show()