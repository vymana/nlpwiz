from sklearn.cluster import AffinityPropagation

from nlpwiz.similarity.similarity import cosine_similarity

DEFAULT_SIMILARITY_MEASURE = cosine_similarity

def cluster_affinity_propagation(texts, similarity_measure=DEFAULT_SIMILARITY_MEASURE):
    """
    Text clustering with  cosine_similarity
    """
    affinity_matrix = similarity_measure(texts)
    affprop = AffinityPropagation(affinity='precomputed', damping=0.5, convergence_iter=5,  max_iter=100,)
    affprop.fit(affinity_matrix)
    return affprop.labels_


if __name__ == "__main__":
    from nlpwiz.clustering import affinity_propagation
    texts = [
        "PUBG Mobile Pro League (PMPL) South Asia 2020: Table-toppers ORANGE ROCK win 1st Chicken Dinner of week 2 ",
        "PUBG Mobile Pro League (PMPL) South Asia 2020: IND pip Marcos Gaming to claim 2nd Chicken Dinner of week 2",
        "PUBG Mobile Pro League (PMPL) South Asia 2020 Week 2, Day 1: SynerGE win Chicken Dinner in Match 5",
        "Donald Trump calls Donnie Drago grossly incompetent person",
        "Donnie Drago was a grossly incompetent person, says Donald Trump",
        "Donald Trump calls Donnie grossly incompetent person "
    ]
    clusters = affinity_propagation.cluster_affinity_propagation(texts)
    #clusters = [0, 0, 0, 1, 1, 1]
