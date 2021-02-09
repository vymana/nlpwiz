
from nlpwiz.clustering import affinity_propagation

def text_clusters():
    texts = [
        "PUBG Mobile Pro League (PMPL) South Asia 2020: Table-toppers ORANGE ROCK win 1st Chicken Dinner of week 2 ",
        "PUBG Mobile Pro League (PMPL) South Asia 2020: IND pip Marcos Gaming to claim 2nd Chicken Dinner of week 2",
        "PUBG Mobile Pro League (PMPL) South Asia 2020 Week 2, Day 1: SynerGE win Chicken Dinner in Match 5",
        "Donald Trump calls Donnie Drago grossly incompetent person",
        "Donnie Drago was a grossly incompetent person, says Donald Trump",
        "Donald Trump calls Donnie grossly incompetent person "
    ]
    clusters = affinity_propagation.cluster_affinity_propagation(texts)
    assert clusters[0] == clusters[1]  == clusters[2]
    assert clusters[3] == clusters[4] == clusters[5]
