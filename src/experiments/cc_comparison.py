import numpy as np
import pandas as pd
from iac import IAC
from modification import UACM, MWCM
import skquery.pairwise as skq
from skquery.oracle import MLCLOracle
from experiments.exp_utils import load_dataset, start_partition
from utils import transitive_closure
from time import time
import active_semi_clustering as asc


def comparison(args, battery, dataset_name, i, algo, active_name):
    """

    Parameters
    ----------
    args : Namespace
        Parsed input arguments.
    battery : str
        Name of the dataset battery.
    dataset_name : str
        Name of the dataset to test.
    i : int
        ID of the experimental run.
    algo : str
        Name of the constrained clustering algorithm to use.
    active_name : str or object
        Name of the active query strategy to use.

    Returns
    -------
    Results are written in text files according to the type of measure.
    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    times = pd.DataFrame()
    l_time = []
    start = start_partition(args, battery, dataset_name)
    if type(active_name) is str:
        active = getattr(skq, active_name)
    else:
        active = active_name
    if algo == "IAC":
        # Interactive clustering loop
        framework = IAC(pd.DataFrame(dataset))
        framework.init_loop(start)
        n = None
        while framework.ask_for_termination(args.iter, auto=args.auto) != "y":
            selector = active(n) if active_name == "NPU" else active()
            framework.select_constraints(selector, MLCLOracle(budget=10, truth=labels))
            _, t = framework.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=0.3))
            l_time.append(t)
            n = selector.neighborhoods if active_name == "NPU" else None

        framework.get_partitions(f"{args.o}/comparison/{dataset_name}/raw/partitions{i+1}_{algo}_{active_name}")
        framework.get_constraints(f"{args.o}/comparison/{dataset_name}/raw/constraints{i+1}_{algo}_{active_name}")
    else:
        alg = getattr(asc, algo)
        dataset = dataset.to_numpy()
        partitions = {0: start}
        l_constraints = []
        n = None
        for j in range(args.iter):
            selector = active(n) if active_name == "NPU" else active()
            constraints = selector.fit(dataset, partitions[j], MLCLOracle(budget=10, truth=labels))
            constraints, _ = transitive_closure(constraints, len(set(partitions[j])))
            l_constraints.append(constraints)
            try:
                t = time()
                cop = alg(n_clusters=len(set(partitions[j])))
                cop.fit(dataset, ml=constraints["ml"], cl=constraints["cl"])
                l_time.append(time() - t)
                partitions[j+1] = cop.labels_
            except Exception as e:
                #if COP fails to find a solution, keep current partition as is and retry at next iteration
                partitions[j+1] = partitions[j]
                l_time.append(np.NaN)
            n = selector.neighborhoods if active_name == "NPU" else None

        pd.DataFrame(partitions).to_csv(f"{args.o}/comparison/{dataset_name}/raw/partitions{i+1}_{algo}_{active_name}.csv")
        get_constraints(l_constraints, f"{args.o}/comparison/{dataset_name}/raw/constraints{i+1}_{algo}_{active_name}")
    times[i+1] = l_time
    times.to_csv(f"{args.o}/comparison/{dataset_name}/raw/times{i+1}_{algo}_{active_name}.csv")


def get_constraints(csts, filename):
    """
    Write a set of constraints in a text file.

    Parameters
    ----------
    csts : dict of lists
        Constraint set.
    filename : str
        Name of file to write.

    Returns
    -------

    """
    res = ""
    for cst_set in csts:
        for key in cst_set:
            for cst in cst_set[key]:
                match key:
                    case "label":
                        res += f"{cst[0]}, {cst[1]}\n"
                    case "ml":
                        res += f"{cst[0]}, {cst[1]}, 1\n"
                    case "cl":
                        res += f"{cst[0]}, {cst[1]}, -1\n"
                    case "triplet":
                        res += f"{cst[0]}, {cst[1]}, {cst[2]}, 3\n"
        res += "\n"

    with open(f"{filename}.txt", "w") as file:
        file.write(res)
