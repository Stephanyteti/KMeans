import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

class KMeans3D:
    
    def _init_(self, data, k, max_iterations=1000):
        self.data = data
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
    
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self):
        # Inicializa os centroides aleatoriamente
        self.centroids = self.data.sample(self.k)

        # Executa o algoritmo de k-means
        for i in range(self.max_iterations):
            # Calcula a distância de cada ponto aos centroides
            distances = pd.DataFrame()
            for j in range(self.k):
                distances[j] = self.data.apply(lambda row: self.euclidean_distance(row, self.centroids.iloc[j]), axis=1)

            # Atribui cada ponto ao cluster mais próximo
            self.labels = distances.idxmin(axis=1)

            # Atualiza os centroides
            new_centroids = self.data.groupby(self.labels).mean()

            # Verifica se houve convergência
            if self.centroids.equals(new_centroids):
                break

            self.centroids = new_centroids

    def plot3d(self, x_label='PC1', y_label='PC2', z_label='PC3'):
        fig = px.scatter_3d(self.data, x='PC1', y='PC2', z='PC3', color=self.labels.astype(str))
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label))
        for i in range(self.k):
            centroid = self.centroids.iloc[i]
            fig.add_trace(px.scatter_3d(x=[centroid[0]], y=[centroid[1]], z=[centroid[2]]).data[0])
        fig.show()

    def plot1_2(self, x_label='PC1', y_label='PC2'):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, 'PC1'], self.data.loc[self.labels == i, 'PC2'], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 0], self.centroids.iloc[i, 1], c=colors[i], marker='x', s=100)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()

    def plot1_3(self, x_label='PC1', y_label='PC3'):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, 'PC1'], self.data.loc[self.labels == i, 'PC3'], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 0], self.centroids.iloc[i, 2], c=colors[i], marker='x', s=100)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()

    def plot2_3(self, x_label='PC2', y_label='PC3'):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, 'PC2'], self.data.loc[self.labels == i, 'PC3'], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 1], self.centroids.iloc[i, 2], c=colors[i], marker='x', s=100)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()