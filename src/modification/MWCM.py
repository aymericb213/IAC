import numpy as np
import pandas as pd
from cpmpy import *
from cpmpy.expressions.globalfunctions import Count
from cpmpy.expressions.variables import cpm_array
from cpmpy.solvers import CPM_ortools, param_combinations
from cpmpy.tools import ParameterTuner
from utils import timer


class MWCM:
    """
    COP modeling the Minimal Weighted Clustering Modification problem :
    find the partition that is most similar to the input while satisfying user constraints.

    Attributes
    ----------
        constrained_partition : list of int
            Subset of the input partition containing the constrained instances (or super-instances, depending on preprocessing).
        k : int
            Number of clusters in the input partition.
        k_max : int
            Domain size of the variables ; if k_max > k, the model is allowed to create clusters.
        p_dists_reps : matrix of float
            Distance matrix linking constrained instances to
            their closest representatives of each cluster (anchors or medoids).
        constraints : dict
            Dictionary of constraints.
        sat_rate : float
            The lowest proportion of constraints the model must satisfy.
        modifications : dict
            Output of the model ; keys are instance indexes, values are tuples
            that contain the previous and new label of the instance.
    """

    def __init__(self, distance_matrix, constrained_partition, constraints, sat_rate=1, k_supp=0):
        self.constrained_partition = constrained_partition
        self.k = len(distance_matrix.columns)
        self.k_max = self.k + k_supp
        self.p_dists_reps = pd.DataFrame(distance_matrix).multiply(100).astype(int)
        self.constraints = constraints
        self.sat_rate = sat_rate
        self.modifications = {}

    @timer
    def solve(self, verbose=False):
        """
        Defines the CSP and solves it.

        Returns
        -------
        None
            The modifications made by the model are stored in the modifications attribute.
        """
        # print(f"{self.k}-clustering")
        new_partition, cst_satisfaction, b, v = self.cop_vars()

        while self.p_dists_reps.shape[1] < self.k_max:
            # create columns for potential new clusters
            # with distances to it greater than to existing clusters
            # so as not to mess up the objective function
            self.p_dists_reps[self.p_dists_reps.shape[1] + 1] = [np.max(self.p_dists_reps.loc[x]) + 1 for x in self.p_dists_reps.index]

        model = Model()

        model, constraint_count = self.user_constraints(model, new_partition, cst_satisfaction)

        # Constraint relaxation : satisfy at least a number of constraints defined by sat rate
        model += sum(cst_satisfaction) >= int(np.ceil(self.sat_rate * constraint_count))

        # Boolvars indicate which instances were modified
        model += (self.constrained_partition != new_partition) == b
        # Weighting the modifications
        for i in range(len(self.constrained_partition)):
            dists_to_reps = cpm_array(self.p_dists_reps.iloc[i])
            model += dists_to_reps[new_partition[i]] == v[i]  # Element global constraint
        # Objective function : minimize distances of modified instances to their new clusters
        model.minimize(sum(b * v))

        #params = tune_solver(model)
        params = {}

        # Solve the model
        s = SolverLookup.get("ortools", model)
        s.solution_hint(new_partition, self.constrained_partition)  # warm starting with current partition
        if s.solve(**params):
            result = np.asarray(new_partition.value())
            if verbose:
                print(f"Start partition : {self.constrained_partition}")
                print(f"Final partition : {result}")
                print(f"Cost = {model.objective_value()}")
            self.get_relaxed_csts(cst_satisfaction)
            for i in range(len(self.constrained_partition)):
                if self.constrained_partition[i] != result[i]:
                    self.modifications[self.p_dists_reps.index[i]] = (self.constrained_partition[i], result[i])
            return self.modifications
        else:
            print("No solution found")
            return {}

    def cop_vars(self):
        """
        Defines problem variables.
        For convenience, all variables are stored in lists
        even when there is only one variable of a kind.

        Returns
        -------
        new_partition : list of intvar
            Variables encoding the new assignment of instances to clusters.
        cst_satisfaction : list of boolvar
            Variables encoding the satisfaction of the constraint by the new partition.
        b : list of boolvar
            Variables indicating the modification of the cluster assignment of an instance.
        v: list of intvar
            Variables encoding the weight of the modification.
        """
        if len(self.constrained_partition) == 1:
            new_partition = [intvar(0, self.k_max - 1, name="X")]
            b = [boolvar(name="B")]
            v = [intvar(0, np.matrix(self.p_dists_reps).max(), name="V")]
        else:
            new_partition = intvar(0, self.k_max - 1, shape=len(self.constrained_partition), name="X")
            b = boolvar(shape=len(self.constrained_partition), name="B")
            v = intvar(0, np.matrix(self.p_dists_reps).max(), shape=len(self.constrained_partition), name="V")

        if sum([len(v) for v in self.constraints.values()]) == 1:
            cst_satisfaction = [boolvar(name="S")]
        else:
            cst_satisfaction = boolvar(shape=sum([len(self.constraints[key]) for key in self.constraints]), name="S")

        return new_partition, cst_satisfaction, b, v

    def user_constraints(self, model, new_partition, cst_satisfaction, verbose=False):
        """
        Adds user constraints to the model.

        Parameters
        ----------
        model : object
            COP model.
        new_partition : list of intvar
            Variables encoding the new assignment of instances to clusters.
        cst_satisfaction : list of boolvar
            Variables encoding the satisfaction of the constraint by the new partition.
        verbose : boolean
            Controls debug print in console.

        Returns
        -------
        model : object
            COP model with user constraints.
        constraint_count : int
            Number of user constraints added to the model.
        """
        constraint_count = 0
        # degree_list = np.array([ct[i] for ct in constraints for i in [0, 1]])
        for key in self.constraints:
            for constraint in self.constraints[key]:
                sat = cst_satisfaction[constraint_count]
                match key.lower():
                    case "label":
                        model += (new_partition[constraint[0]] == constraint[1]) == sat
                    case "ml":
                        model += (new_partition[constraint[0]] == new_partition[constraint[1]]) == sat
                    case "cl":
                        model += (new_partition[constraint[0]] != new_partition[constraint[1]]) == sat
                    case "triplet":
                        # ML(a,n) -> ML(a,p)
                        model += ((new_partition[constraint[0]] == new_partition[constraint[2]]).implies(new_partition[constraint[0]] == new_partition[constraint[1]])) == sat
                        # CL(a,p) -> CL(a,n)
                        # model += ((new_partition[constraint[0]] != new_partition[constraint[1]]).implies(new_partition[constraint[0]] != new_partition[constraint[2]])) == sat
                    case "span":  # ((a,b...n), {k1, k2...kn})
                        """
                        vars = boolvar(shape=(self.k, len(constraint[0])))
                        for c in range(self.k):
                            for i in range(len(constraint[0])):
                                model += (new_partition[constraint[0][i]] == c) == vars[c, i]
                        """
                        group = [new_partition[idx] for idx in constraint[0]]
                        if type(constraint[1]) == set:  # specific case
                            for c in range(self.k):
                                if c in constraint[1]:
                                    pass
                                    # model += sum(vars[c]) >= 0
                                    model += (Count(group, c) > 0) == sat
                                else:
                                    # model += sum(vars[c]) == 0
                                    model += (Count(group, c) == 0) == sat
                            print(model.constraints)
                        else:  # generic case
                            nb_clust = boolvar(shape=self.k)
                            for c in range(self.k):
                                # model += (sum(vars[c]) > 0) == nb_clust[c]
                                model += (Count(group, c) > 0) == nb_clust[c]
                            # model += sum(nb_clust) <= constraint[1]
                            model += (Count(nb_clust, 1) <= constraint[1]) == sat
                    case "impl":
                        model += (new_partition[constraint[0]] == new_partition[constraint[1]]).implies(new_partition[constraint[2]] == new_partition[constraint[3]])
                    case _:
                        raise NotImplementedError(f"Constraint type {key} not recognized")
                constraint_count += 1

        if verbose:
            print(f"{constraint_count} constraints")
            for key in self.constraints:
                print(f"{len(self.constraints[key])} {key}")
        return model, constraint_count

    def get_relaxed_csts(self, cst_satisfaction, verbose=False):
        """
        Retrieve relaxed constraints, if applicable.

        Parameters
        ----------
        cst_satisfaction : list of boolvar
            Variables encoding the satisfaction of the constraint by the new partition.
        verbose : boolean
            Controls debug print in console.

        Returns
        -------
        relaxed_csts : dict of list
            Dictionary of relaxed constraints.
        """
        relaxed_csts = {}
        if (len(cst_satisfaction) == 1 and not cst_satisfaction[0].value()) or (1 < len(cst_satisfaction) != sum(cst_satisfaction.value())):
            relaxed = np.where(cst_satisfaction.value() is False)[0]
            i = 0
            for key in self.constraints:
                for constraint in self.constraints[key]:
                    if i in relaxed:
                        if key not in relaxed_csts:
                            relaxed_csts[key] = []
                        relaxed_csts[key].append(constraint)
                    i += 1
        if verbose:
            print(f"Constraints satisfied: {(1 - (len(relaxed_csts)/sum([len(v) for v in self.constraints.values()]))) * 100}%")
        return relaxed_csts


def tune_solver(model, verbose=False):
    """
    Tune solver parameters according to the model.

    Parameters
    ----------
    model : object
        COP model.
    verbose : boolean
        Controls debug print in console.

    Returns
    -------
    Best values for the parameters tuned w.r.t. the problem.
    """
    tuner = ParameterTuner("ortools", model)
    tuner.tune(max_tries=10)
    if verbose:
        print(tuner.all_params, tuner.best_params, tuner.base_runtime, tuner.best_runtime)
    return tuner.best_params
