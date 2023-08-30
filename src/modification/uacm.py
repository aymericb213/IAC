import numpy as np
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import os
from utils import *
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy


class UACM:
    def __init__(self, model, objective_rate=0, generalization_rate=1, sat_rate=1):
        # defaults to no additional clusters, all constraints satisfied
        self.dataset = None
        self.model = model

        self.k = None
        self.partition = []
        self.representatives = []

        self.objective_rate = objective_rate
        self.sp_gen_rate = generalization_rate
        self.list_sp = {}

        self.constraints = {}
        self.modifications = {}

        self.sat_rate = sat_rate

    def update(self, dataset, partition, true_k, constraints):
        """

        Parameters
        ----------
        dataset
        partition
        true_k
        constraints

        Returns
        -------

        """
        self.dataset = dataset
        self.partition = partition
        self.k = len(set(partition))
        self.representatives, t_anchors = compute_anchors(self.dataset, self.partition, self.objective_rate) if self.objective_rate > 0 else compute_medoids(self.dataset, self.partition)
        self.constraints = constraints
        # print(constraints)
        # self._view_partition_with_anchors()
        _, t_sp = self._superpoints()

        distance_matrix = self._objective_distance_matrix()
        constrained_partition = self._sp_partition() if self.sp_gen_rate < 1 else [partition[i] for i in distance_matrix.index]
        m_constraints = self._generate_model_constraints()

        # self._view_sp_partition(self.list_sp, m_constraints)
        # print(f"Informativeness : {self._informativeness()}")
        # self.plot()
        conflicts = conflict_detection(m_constraints)
        size = sum([len(m_constraints[key]) for key in m_constraints])
        #print(f"{size} constraints")
        self.sat_rate = ((size - conflicts) / size) if self.sat_rate == 1 else self.sat_rate
        self.modifications, t_mod = self.model(distance_matrix, constrained_partition, m_constraints, self.sat_rate, max(0, true_k - self.k)).solve(verbose=True)

        self._generalize_modifications()
        print(f"{len(self.modifications)} instances reaffected ({round((len(self.modifications) / len(self.partition)) * 100, 2)}% of data)")
        print(self.modifications)

        if len(set(self.partition)) < self.k:  # if a cluster has been removed by modifs
            self._reindex_partition()

        # self._view_partition_with_anchors()
        t = t_anchors + t_sp + t_mod
        print(f"Modification within {t} seconds ({t_anchors} generating anchors, {t_sp} generating SI, {t_mod} for COP)")
        return self.partition, self.modifications, t

    def _get_constrained_points(self):
        constrained_points = set()
        for key in self.constraints:
            for cst in self.constraints[key]:
                if key == "span":
                    for x in cst[0]:
                        constrained_points.add(x)
                else:
                    for i in range(len(cst)):
                        constrained_points.add(cst[i])
        return sorted(list(constrained_points))

    def _get_constrained_superpoints(self):
        constrained_sp = set()
        for cluster in self.list_sp:
            for sp_id in self.list_sp[cluster]:
                for x in self.list_sp[cluster][sp_id]:
                    if [ct for ct_list in self.constraints.values() for ct in ct_list if x in ct]:
                        constrained_sp.add(sp_id)
        return sorted(list(constrained_sp))

    def _generate_model_constraints(self):
        # Reindexing constraints based on the set of constrained instances
        # to limit the size of the CSP
        model_constraints = {}
        instances = self._get_constrained_points()
        if self.sp_gen_rate < 1:
            s_instances = self._get_constrained_superpoints()
        for key in self.constraints:
            model_constraints[key] = []
            for ct in self.constraints[key]:
                if self.sp_gen_rate < 1:
                    corresponding = [s_instances.index(sp_id) for x in ct for sp_id in self.list_sp[self.partition[x]] if x in self.list_sp[self.partition[x]][sp_id]]
                else:
                    if key == "span":
                        corresponding = [instances.index(x) for x in ct[0]]
                    else:
                        corresponding = [instances.index(x) for x in ct]
                match key:
                    case "label":
                        model_constraints[key].append((corresponding[0], ct[1]))
                    case "ml" | "cl" | "triplet":
                        model_constraints[key].append(tuple(corresponding))
                    case "span":
                        model_constraints[key].append((tuple(corresponding), ct[1]))
                    case _:
                        raise ValueError(f"{key} No corresponding instances found for the model : {ct}")
        return model_constraints

    @timer
    def _superpoints(self):
        if self.sp_gen_rate == 1.:
            return

        self._clustering_superpoints()
        # self._nearest_neighbors_superpoints()

    def _nearest_neighbors_superpoints(self):
        si_comp = []

        for c in set(self.partition):
            cluster = np.where(self.partition == c)[0]

            nn = NearestNeighbors(n_neighbors=len(cluster) // self.sp_gen_rate, algorithm="ball_tree")
            nn.kneighbors(self.dataset.iloc[cluster])

    def _clustering_superpoints(self):
        # Generation
        si_comp = {}  # index is cluster number, each dict links a super-instance number to its components

        for c in set(self.partition):
            cpt = sum([len(si_comp[key]) for key in si_comp])

            cluster = np.where(self.partition == c)[0]
            if len(cluster) == 1:
                si_comp[c] = {cpt: [cluster[0]]}
            else:
                cluster_si = {}
                nb_si = max(1, int(len(cluster) * self.sp_gen_rate))

                split = AgglomerativeClustering(n_clusters=nb_si, linkage="complete") if len(cluster) < 30000 else OPTICS(min_samples=len(cluster) // nb_si, n_jobs=-1)
                split.fit_predict(self.dataset.iloc[cluster])
                for i in range(nb_si):
                    cluster_si[cpt + i] = [cluster[j] for j in range(len(cluster)) if split.labels_[j] == i]
                si_comp[c] = cluster_si

        # self._view_sp_partition(si_comp, False)

        # Constraint consistency
        cpt = sum([len(si_comp[key]) for key in si_comp])
        constrained = self._get_constrained_points()
        for key in si_comp:
            to_remove = {}
            for si in deepcopy(si_comp[key]):
                candidates = list(set(constrained).intersection(set(si_comp[key][si])))
                if len(candidates) > 1:  # multiple constrained instances in same SI
                    cand_sp_map = {}  # mapping between new super-instances and their ids for reaffectation
                    to_remove[si] = []
                    for x in candidates[1:]:
                        cand_sp_map[x] = cpt
                        si_comp[key][cpt] = [x]
                        to_remove[si].append(x)
                        cpt += 1

                    dsts = pd.DataFrame(pairwise_distances(self.dataset.iloc[si_comp[key][si]], self.dataset.iloc[candidates]), index=si_comp[key][si], columns=candidates)

                    for i in range(len(si_comp[key][si])):
                        if si_comp[key][si][i] not in dsts.columns:
                            cand = dsts.columns[np.argmin(dsts.iloc[i, :])]  # closest centroid
                            if cand != candidates[0]:
                                si_comp[key][cand_sp_map[cand]].append(si_comp[key][si][i])
                                to_remove[si].append(si_comp[key][si][i])
            for si in to_remove:
                for pt in to_remove[si]:
                    si_comp[key][si].remove(pt)

        self.list_sp = si_comp

    def _view_sp_partition(self, si_comp, splitted=None):
        viz_dataset = PCA(n_components=2).fit_transform(self.dataset) if self.dataset.shape[1] > 3 else self.dataset
        si_partition = np.zeros(len(self.partition))
        weights = np.ones(len(si_partition))
        s_instances = self._get_constrained_superpoints()
        if splitted:
            for i in range(len(si_partition)):
                si_partition[i] = -1

            for cluster in si_comp:
                for si in si_comp[cluster]:
                    if si in s_instances:
                        for x in si_comp[cluster][si]:
                            si_partition[x] = si
                            weights[x] = 5

        else:
            for cluster in si_comp:
                for si in si_comp[cluster]:
                    for x in si_comp[cluster][si]:
                        si_partition[x] = si

        si_partition = si_partition.astype(str)

        fig = px.scatter(viz_dataset, x=0, y=1, template="simple_white",
                         color=si_partition, symbol=si_partition, size=weights,
                         color_discrete_map={"-1.0": "gray"})
        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        for key in self.constraints:
            for cst in self.constraints[key]:
                points = viz_dataset.iloc[list(cst)]
                fig.add_trace(go.Scatter(name=str(cst), x=[points.iloc[0, 0], points.iloc[1, 0]],
                                         mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]]))
                if key == "ml":
                    fig['data'][-1]['line']['color'] = "#ff0000"
                else:
                    fig['data'][-1]['line']['color'] = "#0000ff"
                    fig['data'][-1]['line']['dash'] = "dash"
        b = True if splitted else False
        fig.write_html(f"sp_partition_{self.sp_gen_rate}_{b}.html")

    def _sp_partition(self, constrained=True):
        if constrained:
            constrained_si = self._get_constrained_superpoints()
            partition = np.zeros(len(constrained_si))
            for x in self.list_sp:
                for sp_id in self.list_sp[x]:
                    if sp_id in constrained_si:
                        partition[constrained_si.index(sp_id)] = x
        else:
            nb_sp = sum([len(dicts) for dicts in self.list_sp])
            partition = np.zeros(nb_sp)
            for x in self.list_sp:
                for sp_id in self.list_sp[x]:
                    partition[sp_id] = x
        return partition.astype(int)

    def _objective_distance_matrix(self):
        points = self._get_constrained_superpoints() if self.sp_gen_rate < 1 else self._get_constrained_points()
        # matrix where the distance to the closest anchor of each cluster is stored for each constrained instance
        matrix = pd.DataFrame()
        for cp in points:
            if self.sp_gen_rate < 1:
                clust_id = next(cl for cl in set(self.partition) if cp in self.list_sp[cl])  # gets cluster where super-instance cp belongs
            anchor_row = []
            for c in set(self.partition):
                if self.sp_gen_rate < 1:
                    df = pd.DataFrame(pairwise_distances(self.dataset.iloc[self.representatives[c]], self.dataset.iloc[self.list_sp[clust_id][cp]]), index=self.representatives[c],
                                      columns=self.list_sp[clust_id][cp]).mean(axis=1)
                    anchor_row.append(df.min())
                else:
                    df = pd.DataFrame(pairwise_distances(self.dataset.iloc[self.representatives[c]], np.reshape(self.dataset.iloc[cp].array, (1, -1))), index=self.representatives[c], columns=[cp])
                    anchor_row.append(df.min()[cp])
            matrix = pd.concat([matrix, pd.DataFrame(anchor_row, columns=[cp]).T])
        # print(matrix)
        return matrix

    def _view_partition_with_anchors(self):
        print(self.representatives)
        viz_dataset = PCA(n_components=2).fit_transform(self.dataset) if self.dataset.shape[1] > 3 else self.dataset
        weights = np.ones(len(self.dataset))
        for clust_dict in self.representatives:
            for anchor in self.representatives[clust_dict]:
                weights[anchor] = 10
        fig = px.scatter(viz_dataset, x=0, y=1, template="simple_white", size=weights,
                         color=self.partition.astype(str), symbol=self.partition.astype(str))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.write_html(f"partition_{self.objective_rate}.html")

    def _informativeness(self):
        count = 0
        size = 0
        for key in self.constraints:
            size += len(self.constraints[key])
            for cst in self.constraints[key]:
                if key == "label":
                    if self.partition[cst[0]] != cst[1]:
                        count += 1
                elif key == "ml":
                    if self.partition[cst[0]] != self.partition[cst[1]]:
                        count += 1
                elif key == "cl":
                    if self.partition[cst[0]] == self.partition[cst[1]]:
                        count += 1
        # informativeness = proportion of constraints unsatisfied by the base partition
        return count / size

    def _generalize_modifications(self, mode="super"):
        """
        Propagates the changes made by the CSP to the partition using one of 4 modes.
            @param mode: how the modifications are propagated. Can be one of "" (no propagation),
            "super" (super-instances) "knn" (nearest neighbors) or "radius"
        """
        match mode:
            case "knn":
                self.modifications.update(self._knn_propagation())
            case "radius":
                self.modifications.update(self._distance_propagation())
            case "super" if self.sp_gen_rate < 1:
                self.modifications = self._sp_propagation()
            case _:
                for x in self.modifications:
                    self.partition[x] = self.modifications[x][1]

    def _knn_propagation(self, knn=5):
        """
        k-nearest neighbors propagation. CSP changes are propagated
        to the nearest neighbors of each modified instance,
        according to Euclidean distance

        Parameters
        ----------
        knn : int
            number of neighbors to which the changes must be propagated

        Returns
        -------
            A dictionary whose keys are the propagated instances and values are tuples of form (old_cluster, new_cluster)
        """
        propagations = {}
        c_dists = pd.DataFrame(pairwise_distances(self.dataset, self.dataset.iloc[list(self.modifications.keys())]), columns=list(self.modifications.keys()))

        for x in self.modifications:
            neighbors = set(c_dists[x].nsmallest(n=knn + 1, keep="all").index.values)

            for supp in neighbors:
                if supp in self.modifications.keys() and supp != x:
                    pass  # don't change an instance already modified within CSP
                else:
                    propagations[supp] = (self.partition[supp], self.modifications[x][1])
                    self.partition[supp] = self.modifications[x][1]
        return propagations

    def _distance_propagation(self, radius=0.05):
        propagations = {}
        c_dists = pd.DataFrame(pairwise_distances(self.dataset, self.dataset.iloc[list(self.modifications.keys())]), columns=list(self.modifications.keys()))

        for x in self.modifications:
            neighbors = set(c_dists[c_dists[x] <= radius].index.values)

            for supp in neighbors:
                if supp in self.modifications.keys() and supp != x:
                    pass  # don't change an instance already modified within CSP
                else:
                    propagations[supp] = (self.partition[supp], self.modifications[x][1])
                    self.partition[supp] = self.modifications[x][1]
        return propagations

    def _sp_propagation(self):
        propagations = {}
        for instance in self.modifications.keys():
            points = self.list_sp[self.modifications[instance][0]][instance]  # get the points represented by the super-instance
            for x in points:
                propagations[x] = (self.partition[x], self.modifications[instance][1])
                self.partition[x] = self.modifications[instance][1]
        return propagations

    def _reindex_partition(self):
        for clust_id in set(self.partition):
            if clust_id > 0 and clust_id - 1 not in set(self.partition):
                for x in np.where(self.partition == np.max(self.partition))[0]:
                    self.partition[x] = clust_id - 1
