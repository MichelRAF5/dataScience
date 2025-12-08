#%%
# Criando a exibição gráfica de dados fictícios

#%% 

import matplotlib.pylab as plt

dias= list(range(1, 11))
temperaturas= [66.6, 55.8, 35, 70, 80.8, 99, 97.8, 59.4, 49.9, 86.2]

plt.plot(dias, temperaturas, marker= 'o')
plt.title("Temperatura da máquinas em ºC")
plt.xlabel("Dia")
plt.ylabel("Temperatura")
plt.grid(True)
plt.show()