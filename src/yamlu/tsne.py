import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def tsne_scatter(X, Y, perplexity=40, n_iter=400):
    """
    inspired by: https://www.datacamp.com/community/tutorials/introduction-t-sne
    """
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X)

    num_classes = len(np.unique(Y))

    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes))

    le = LabelEncoder()
    y_le = le.fit_transform(Y)

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax: plt.Axes = plt.subplot(aspect='equal')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], lw=0, s=40, c=palette[y_le.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(X_tsne[y_le == i, :], axis=0)
        txt = ax.text(xtext, ytext, le.inverse_transform([i])[0], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax


def tsne_pca_scatter(X, Y, pca_dims=50):
    pca = PCA(n_components=pca_dims)
    X_pca = pca.fit_transform(X)
    tsne_scatter(X_pca, Y)
