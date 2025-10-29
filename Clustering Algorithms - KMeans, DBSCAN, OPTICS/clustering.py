import pandas as pd
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors

# load data
df = pd.read_excel("./Clustering Algorithms - KMeans, DBSCAN, OPTICS/Data/Test_GeneExpressionData.xlsx")
df.columns
df.shape

# number of unique cell types: 10
df['Coded cell type'].unique()

# separate out predictors
X = df.copy()
X = X.drop(columns=['Coded cell type'])

# standardize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(df_scaled, columns=X.columns)


#####################
## KMeans 
# explore potential values of K
inertias = []
k_range = range(1,15)
for i in k_range:
    kmeans = KMeans(n_clusters=i, random_state=1989)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(k_range, inertias, marker='o')### 
plt.title('Elbow method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/k_means_elbow_plot.png")
plt.show()

# elbow plot above bends at k=9
kmeans = KMeans(n_clusters=9, random_state=1989)
kmeans.fit(df_scaled)
# get lables for the cluster assignments 
labels = kmeans.labels_

# PCA for visualization... I know better now, but this was a class assignment
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualization
plt.figure()
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('Cluster Visualization with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/k_means_clusters.png")
plt.show()


#######################
## DBSCAN
## Look at a few different minimum samples values
min_samp = [3, 5, 7, 9]
for i in min_samp:
    min_samples = i
    # define nn as size of neighborhood
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(df_scaled)
    # extract the distance for how far each point is from the edge of the neighborhood
    distances, indices = neighbors_fit.kneighbors(df_scaled)
    # sort in ascending order
    distances = np.sort(distances[:, -1])
    # plot
    plt.figure()
    plt.plot(distances)
    plt.title('Elbow plot for DBSCAN')
    plt.xlabel('Points sorted by distance to {} nearest neighbor'.format(min_samples))
    plt.ylabel('Distance to {} nearest neighbor'.format(min_samples))
    plt.show()

# decide on a min_samp value
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(df_scaled)
distances, indices = neighbors_fit.kneighbors(df_scaled)
distances = np.sort(distances[:, -1])
# plot
plt.figure()
plt.plot(distances)
plt.title('Elbow plot for DBSCAN')
plt.xlabel('Points sorted by distance to {}th nearest neighbor'.format(min_samples))
plt.ylabel('Distance to {}th nearest neighbor'.format(min_samples))
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/distance_to_nearest_neighbor.png")
plt.show()

# define dbscan params and fit
db = DBSCAN(eps=4.5, min_samples=min_samples).fit(df_scaled)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
dbscan_labels = db.labels_

# pca again
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# plot
plt.figure(figsize=(10,8))
unique_labels = set(dbscan_labels)
colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (dbscan_labels == k)
    plt.scatter(df_pca[class_member_mask, 0],
                df_pca[class_member_mask, 1],
                c=[col], edgecolor='k', s=50, label=f'Cluster {k}')
plt.title(f'DBSCAN clusters (n={n_clusters_})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/dbscan_clusters.png")
plt.show()



#################
## OPTICS 
# params
min_samples = 5
xi = 0.01
epsilon = 5.5
cluster_method = 'xi'
metric = 'minkowski'

# define and fit
opt = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, xi=xi, metric=metric).fit(df_pca)
opt_labels = opt.labels_

# get cluster number and noise points
num_clusters = len(np.unique(opt_labels[opt_labels != -1])) # noise == -1
num_noise = np.sum(opt_labels == -1)
print(f'Number of clusters: {num_clusters}')
print(f'Number of noise points: {num_noise}')

# plot
unique_labels = np.unique(opt_labels)
colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'k'
    class_member_mask = (opt_labels == label)
    xy = df_pca[class_member_mask] 
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)
plt.title('OPTICS Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/optics_cluster.png")
plt.show()

# reachability plot
reachability = opt.reachability_[opt.ordering_]
plt.figure()
plt.plot(reachability)
plt.title('Reachability Plot')
plt.xlabel('Sample Index')
plt.ylabel('Reachability Distance')
plt.savefig("Clustering Algorithms - Kmeans, DBSCAN, OPTICS/figs/optics_reacability.png")
plt.show()