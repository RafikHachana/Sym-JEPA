import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def visualize_umap_clusters(vectors, colors, mode='2d', labels=None):
    """
    Visualize 512-dimensional vectors in 2D or 3D using UMAP.

    Parameters:
    - vectors: numpy array of shape (n_samples, 512)
    - colors: list or array of shape (n_samples,) with color values or cluster labels
    - mode: '2d' or '3d'
    """
    assert mode in ['2d', '3d'], "mode must be either '2d' or '3d'"
    vectors = np.array(vectors)

    reducer = UMAP(n_components=2 if mode == '2d' else 3, random_state=42)
    embedding = reducer.fit_transform(vectors)

    fig = plt.figure(figsize=(8, 6))
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=10, alpha=0.8)
        ax.set_title('UMAP Projection (3D)')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.8)
        ax.set_title('UMAP Projection (2D)')
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

    plt.tight_layout()
    plt.show()

def visualize_pca_clusters(vectors, colors, mode='2d'):
    """
    Visualize 512-dimensional vectors in 2D or 3D using PCA.

    Parameters:
    - vectors: numpy array of shape (n_samples, 512)
    - colors: list or array of shape (n_samples,) with color values or cluster labels
    - mode: '2d' or '3d'
    """
    assert mode in ['2d', '3d'], "mode must be either '2d' or '3d'"
    vectors = np.array(vectors)

    reducer = PCA(n_components=2 if mode == '2d' else 3, random_state=42)
    embedding = reducer.fit_transform(vectors)

    fig = plt.figure(figsize=(8, 6))
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=10, alpha=0.8)
        ax.set_title('PCA Projection (3D)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.8)
        ax.set_title('PCA Projection (2D)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    plt.tight_layout()
    plt.show()

def visualize_tsne_clusters(vectors, colors, mode='2d'):
    """
    Visualize 512-dimensional vectors in 2D or 3D using t-SNE.

    Parameters:
    - vectors: numpy array of shape (n_samples, 512)
    - colors: list or array of shape (n_samples,) with color values or cluster labels
    - mode: '2d' or '3d'
    """
    assert mode in ['2d', '3d'], "mode must be either '2d' or '3d'"
    vectors = np.array(vectors)

    reducer = TSNE(n_components=2 if mode == '2d' else 3, random_state=42, perplexity=min(30, len(vectors)-1))
    embedding = reducer.fit_transform(vectors)

    fig = plt.figure(figsize=(8, 6))
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=10, alpha=0.8)
        ax.set_title('t-SNE Projection (3D)')
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
        ax.set_zlabel('t-SNE-3')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.8)
        ax.set_title('t-SNE Projection (2D)')
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')

    plt.tight_layout()
    plt.show()
