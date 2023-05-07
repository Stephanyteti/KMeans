import classe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def main():
#recebe o arquivo do excel, número de clusters e número de iterações
    arquivo = str(input('Digite o local do arquivo(arquivo excel): '))
    k = int(input('Digite o número de clusters(inteiro e positivo): '))
    iteracoes = int(input('Digite o número de iterações(inteiro e positivo): '))
    dados = classe.KMeans3D(pd.read_excel(arquivo),k,iteracoes)

#aplica k-means na base de dados
    dados.fit(3)

#plota os graficos
    dados.plot3d()
    dados.plot1_2()
    dados.plot1_3()
    dados.plot2_3()

if __name__ == '__main__':
    main()