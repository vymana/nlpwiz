import traceback
import numpy as np
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import ward, dendrogram, fcluster

from nlpwiz.similarity import similarity
default_similarity_measure = similarity.cosine_similarity

def cluster_affinity_propagation(texts, similarity_measure=default_similarity_measure):

    affinity_matrix = similarity_measure(texts)

    #affprop = AffinityPropagation(affinity='precomputed', damping=0.5)
    affprop = AffinityPropagation(affinity='precomputed', damping=0.5, convergence_iter=5,  max_iter=100,)
    affprop.fit(affinity_matrix)
    #cluster_centers_indices = af.cluster_centers_indices_
    #labels = af.labels_
    return affprop.labels_

    '''
    affinity_matrix = text_utils.cosine_similarity(all_docs)
    #affprop.fit(affinity_matrix)
    #AffinityPropagation(affinity='precomputed', convergence_iter=15, copy=True,
              damping=0.5, max_iter=200, preference=None, verbose=False)
    affprop = AffinityPropagation(affinity='precomputed', damping=0.5)

    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))
    '''


def cluster_ward_clustering(texts, similarity_measure):
    """
    http://brandonrose.org/clustering
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html

    from scipy.cluster.hierarchy import ward, dendrogram
    import matplotlib.pyplot as plt
    """
    dist = 1-similarity_measure(texts)
    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    clusters = fcluster(linkage_matrix, t=0.6, criterion='inconsistent', depth=2, R=None, monocrit=None)
    return list(clusters)


def cluster_texts(texts):
    labels = cluster_affinity_propagation(texts)
    unique_labels = list(set(labels))
    clustered_texts = [[] for idx in range(len(unique_labels))]

    for idx,cluster_id in enumerate(labels):
        clustered_texts[cluster_id].append(texts[idx])
    return clustered_texts


if __name__ == "__main__":
    pass