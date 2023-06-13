import classe_elbow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def main():
#recebe o arquivo do excel e número de iterações
    arquivo = str(input('Digite o local do arquivo(arquivo excel): '))
    iteracoes = int(input('Digite o número de iterações(inteiro e positivo): '))
    dados = classe_elbow.KMeans3D(pd.read_excel(arquivo),iteracoes)
#aplica o elbow method na base de dados e plota gráfico de cluster x distorção
    dados.elbow_method()
#recebe número de clusters
    while True:
      dados.k = int(input("selecione um valor de k (o valor deve ser entre 1 e 10): "))
      if dados.k > 1 and dados.k < 10:
        break
#aplica k-means na base de dados com k clusters
    dados.fit()
#plota os graficos
    if len(dados.data.columns) == 3:
      dados.plot3d()
      dados.plot1_2()
      dados.plot1_3()
      dados.plot2_3()
    else:
      dados.plot2d()

if __name__ == '__main__':
    main()