import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from itertools import combinations, product


def timer(function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = function(*args, **kwargs)
        t2 = time() - t1
        # print(f"{function.__name__} completed in {t2} seconds")
        return result, t2

    return wrapper


@timer
def compute_anchors(dataset, partition, rate):
    assert len(partition) > 0
    anchors = {}  # cluster ID -> list of anchors

    for c in set(partition):
        cluster = np.where(partition == c)[0]
        if len(cluster) == 1:
            anchors[c] = [cluster[0]]
        elif len(cluster) < 30000:
            # using Single Link
            clustering = AgglomerativeClustering(linkage='single', n_clusters=max(1, int(len(cluster) * rate)))
            clustering.fit_predict(dataset.iloc[cluster])

            # Then we can define the anchors
            cluster_anchors = []
            for k in set(clustering.labels_):
                subcluster = np.where(clustering.labels_ == k)[0]
                sub_index = [dataset.iloc[cluster].index[x] for x in subcluster]  # indexes of subcluster in the original dataset
                dists = pd.DataFrame(pairwise_distances(dataset.iloc[subcluster]), index=sub_index, columns=sub_index).sum()
                cluster_anchors.append(dists.idxmin())  # anchor is instance closest to all other members of subcluster

            anchors[c] = sorted(cluster_anchors)
        elif len(cluster) < 80000:
            # farthest-first traversal
            cluster_anchors = []
            # first anchor
            dists = pd.DataFrame(pairwise_distances(dataset.iloc[cluster], metric="euclidean"), index=cluster, columns=cluster)
            cluster_anchors.append(dists.sum().idxmax())  # furthest point of the cluster (max sum of distances)
            cluster = np.delete(cluster, np.where(cluster == cluster_anchors[-1]))
            # other anchors, up to a fraction of cluster size
            for i in range(int(len(cluster) * rate) - 1):
                head_dists = pd.DataFrame(pairwise_distances(dataset.iloc[cluster], dataset.iloc[cluster_anchors], metric="euclidean"), index=cluster, columns=cluster_anchors)
                cluster_anchors.append(head_dists.sum(axis=1).idxmax())  # furthest point from already selected anchors
                cluster = np.delete(cluster, np.where(cluster == cluster_anchors[-1]))
            anchors[c] = sorted(cluster_anchors)
        else:
            # cluster is too big, only medoid
            centroid = dataset.iloc[cluster].mean()
            medoid = pd.DataFrame(pairwise_distances(dataset.iloc[cluster], pd.DataFrame(centroid).T)).idxmin()[0]
            anchors[c] = [medoid]
    # print("Anchors : {}".format(anchors))
    return anchors


@timer
def compute_medoids(dataset, partition):
    assert len(partition) > 0
    medoids = {}

    for c in set(partition):
        cluster = np.where(partition == c)[0]
        if len(cluster) == 1:
            medoids[c] = [cluster[0]]
        else:
            centroid = dataset.iloc[cluster].mean()
            medoid = pd.DataFrame(pairwise_distances(dataset.iloc[cluster], pd.DataFrame(centroid).T), index=cluster).idxmin()[0]
            medoids[c] = [medoid]
    return medoids


def conflict_detection(constraints):
    conflicts = []
    if "label" in constraints and len(constraints["label"]) > 1:
        for (a, b) in constraints["label"]:
            for (c, d) in constraints["label"]:
                if a == c and b != d:
                    conflicts.append(a)
                    # print(f"Conflicting constraints {(a,b)} and {(c,d)}. Allowing to relax one constraint")
    if "ml" in constraints and "cl" in constraints:
        for (a, b) in constraints["ml"]:
            for (c, d) in constraints["cl"]:
                if (a, b) not in conflicts and a in (c, d) and b in (c, d):
                    conflicts.append((a, b))
                    # print(f"Conflicting constraints {(a,b)} and {(c,d)}. Allowing to relax one constraint")
    # print(f"{len(conflicts)} conflicts detected. Setting sat_rate accordingly")
    return len(conflicts)


def eval_closure(constraints, added, set1, set2, key):
    if len(set1) == 2 and len(set2) == 2:
        common = set1.intersection(set2)
        if len(common) == 1:
            ct = tuple(set1.symmetric_difference(set2))
            if ct not in constraints[key] and len(ct) == 2:
                added[key].add(ct)


@timer
def transitive_closure(constraints, k):
    if "ml" in constraints and "cl" in constraints:
        while True:
            added = {"ml": set(), "cl": set()}
            rule_1 = combinations(constraints["ml"], 2)
            rule_2 = product(constraints["ml"], constraints["cl"])
            for comb in rule_1:  # ML(a,b) and ML(b,c) -> ML(a,c)
                eval_closure(constraints, added, set(comb[0]), set(comb[1]), "ml")
            for comb in rule_2:  # ML(a,b) and CL(b,c) -> CL(a,c)
                eval_closure(constraints, added, set(comb[0]), set(comb[1]), "cl")
            if k == 2:  # binary clustering
                rule_3 = combinations(constraints["cl"], 2)
                for comb in rule_3:  # CL(a,b) and CL(b,c) -> ML(a,c)
                    eval_closure(constraints, added, set(comb[0]), set(comb[1]), "ml")
            if added["ml"] == set() and added["cl"] == set():
                break  # end when no more constraints can be derived
            constraints["ml"] += added["ml"]
            constraints["cl"] += added["cl"]
    return constraints


def is_satisfied(constraint, ct_type, partition):
    match ct_type:
        case "label":
            return partition[constraint[0]] == constraint[1]
        case "ml":
            return partition[constraint[0]] == partition[constraint[1]]
        case "cl":
            return partition[constraint[0]] != partition[constraint[1]]
        case "triplet":
            return (partition[constraint[0]] != partition[constraint[1]] and partition[constraint[0]] == partition[constraint[2]]) or \
                (partition[constraint[0]] == partition[constraint[2]] and partition[constraint[0]] != partition[constraint[1]])
        case "span":
            clusters = set()
            for x in constraint[0]:
                clusters.add(partition[x])
            return (type(constraint[1]) == set and clusters != constraint[1]) or \
                (type(constraint[1]) == int and len(clusters) > constraint[1])


def satisfaction_rate(partition, constraints):
    size = sum([len(constraints[key]) for key in constraints])
    unsat = 0
    for key in constraints:
        for ct in constraints[key]:
            if not is_satisfied(ct, key, partition):
                unsat += 1
    return unsat, (size - unsat) / size


def ind2coord(i, length):
    return i % length, i // length


def coord2ind(p, length):
    return p[1] * length + p[0]
