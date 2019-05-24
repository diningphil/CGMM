import time
import pandas as pd
import seaborn as sns

from sklearn import decomposition
from sklearn.manifold import TSNE


def tsne(data, pca_components=0, no_components=2, perplexity=30, max_no_points=-1):

    embeddings = data
    if max_no_points > 1:
        embeddings = embeddings[:max_no_points, :]

    if pca_components >= 1:
        print('Doing PCA first...')
        pca = decomposition.PCA(n_components=pca_components)

        t = int(time.time())
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)
        print('PCA done. Time elapsed ', int(time.time())-t, 'seconds.')

    print('Doing T-SNE...')
    t = int(time.time())
    embeddings = TSNE(n_components=no_components, perplexity=perplexity).fit_transform(embeddings)
    print('T-SNE done. Time elapsed ', int(time.time()) - t, 'seconds.')

    return embeddings
