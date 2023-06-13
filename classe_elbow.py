import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

class KMeans3D:
    
    def __init__(self, data, k = 10, max_iterations=1000):
        self.data = data
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

    def elbow_method(self, max_clusters=10):
        distortions = []
        for k in range(1, max_clusters + 1):
            self.k = k
            self.fit()
            distortion = 0
            #calcula distorção
            for j in range(self.k):
                cluster_points = self.data[self.labels == j]
                centroid = self.centroids.iloc[j]
                distortion += np.sum((cluster_points - centroid) ** 2)
            distortions.append(distortion)
        
        # Plotar o gráfico de distorção (elbow plot)
        plt.plot(range(1, max_clusters + 1), distortions, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Distortion')
        plt.title('Método Elbow')
        plt.show()


    def plot3d(self):
        fig = px.scatter_3d(self.data, x = self.data.columns[0], y=self.data.columns[1], z=self.data.columns[2], color=self.labels.astype(str))
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(scene=dict(xaxis_title=self.data.columns[0], yaxis_title=self.data.columns[1], zaxis_title=self.data.columns[2]))
        for i in range(self.k):
            centroid = self.centroids.iloc[i]
            fig.add_trace(px.scatter_3d(x=[centroid[0]], y=[centroid[1]], z=[centroid[2]]).data[0])
        fig.show()

    def plot2d(self):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        if len(self.data.columns) == 2:
            for j in range(self.k):
                ax.scatter(self.data.loc[self.labels == j, self.data.columns[0]], self.data.loc[self.labels == j, self.data.columns[1]], c=colors[j], marker='o',s=10)
                ax.scatter(self.centroids.iloc[j, 0], self.centroids.iloc[j, 1], c=colors[j], marker='x', s=100)
            ax.set_xlabel(self.data.columns[0])
            ax.set_ylabel(self.data.columns[1])
            plt.show()
        else:
            for i in range(3):
                if i < 2:
                    for j in range(self.k):
                        ax.scatter(self.data.loc[self.labels == j, self.data.columns[0]], self.data.loc[self.labels == j, self.data.columns[i+1]], c=colors[j], marker='o',s=10)
                        ax.scatter(self.centroids.iloc[j, 0], self.centroids.iloc[j, 1], c=colors[j], marker='x', s=100)
                    ax.set_xlabel(self.data.columns[0])
                    ax.set_ylabel(self.data.columns[i+1])
                    plt.show()
                else:
                    for j in range(self.k):
                        ax.scatter(self.data.loc[self.labels == j, self.data.columns[1]], self.data.loc[self.labels == j, self.data.columns[i]], c=colors[j], marker='o',s=10)
                        ax.scatter(self.centroids.iloc[j, 0], self.centroids.iloc[j, 1], c=colors[j], marker='x', s=100)
                    ax.set_xlabel(self.data.columns[1])
                    ax.set_ylabel(self.data.columns[2])
                    plt.show()
    
    def plot1_2(self):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, self.data.columns[0]], self.data.loc[self.labels == i, self.data.columns[1]], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 0], self.centroids.iloc[i, 1], c=colors[i], marker='x', s=100)
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        plt.show()
    
    def plot1_3(self):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, self.data.columns[0]], self.data.loc[self.labels == i, self.data.columns[2]], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 0], self.centroids.iloc[i, 2], c=colors[i], marker='x', s=100)
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[2])
        plt.show()

    def plot2_3(self):
        # Plota o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b','y','m','black','brown']
        for i in range(self.k):
            ax.scatter(self.data.loc[self.labels == i, self.data.columns[1]], self.data.loc[self.labels == i, self.data.columns[2]], c=colors[i], marker='o',s=10)
            ax.scatter(self.centroids.iloc[i, 1], self.centroids.iloc[i, 2], c=colors[i], marker='x', s=100)
        ax.set_xlabel(self.data.columns[1])
        ax.set_ylabel(self.data.columns[2])
        plt.show()