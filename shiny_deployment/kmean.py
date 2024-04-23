import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

def pull_dataset():
    X, y = load_iris(return_X_y=True)
    data = pd.DataFrame(X)
    data.columns = ['sepal_length','sepal_width', 'petal_length', 'petal_width']
    return data.head()
def plot_elbow():
    X, y = load_iris(return_X_y=True)
    sse = []
    for k in range(1,11):
        km = KMeans(n_clusters=k, random_state=2)
        km.fit(X)
        sse.append(km.inertia_)
    # Set labels and title
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Sum Squared Error")
    ax.set_title("Elbow Method")
    ax.plot(range(1, 11), sse, marker='o', linestyle='-', color='b')
    return fig

def train_kmeans():
    X, y = load_iris(return_X_y=True)
    sse = []
    kmeans = KMeans(n_clusters = 2, random_state = 2)
    kmeans.fit(X)
    kmeans.cluster_centers_
    pred = kmeans.fit_predict(X)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting the first subplot
    ax[0].scatter(X[:, 0], X[:, 1], c=pred, cmap=cm.Accent)
    ax[0].grid(True)
    for center in kmeans.cluster_centers_:
        ax[0].scatter(center[0], center[1], marker='^', c='red')
    ax[0].set_xlabel("petal length (cm)")
    ax[0].set_ylabel("petal width (cm)")

    # Plotting the second subplot
    ax[1].scatter(X[:, 2], X[:, 3], c=pred, cmap=cm.Accent)
    ax[1].grid(True)
    for center in kmeans.cluster_centers_:
        ax[1].scatter(center[2], center[3], marker='^', c='red')
    ax[1].set_xlabel("sepal length (cm)")
    ax[1].set_ylabel("sepal width (cm)")
    return fig
    
def get_coef():
    X, y = load_iris(return_X_y=True)
    sse = []
    kmeans = KMeans(n_clusters = 2, random_state = 2)
    kmeans.fit(X)
    kmeans.cluster_centers_
    pred = kmeans.fit_predict(X)
    return silhouette_score(X, pred)