import numpy as np
import pandas as pd
from modification import UACM, MWCM
from skquery.pairwise import RandomMLCL
from skquery.oracle import MLCLOracle
from iac import IAC
import active_semi_clustering as asc
from experiments.exp_utils import load_dataset, start_partition
import random
from time import time
from utils import satisfaction_rate
from experiments.cc_comparison import get_constraints


def relax_experiment(args, battery, dataset_name, i, algo, conflict_exp):
    """

    Parameters
    ----------
    args
    battery
    dataset_name
    i
    algo
    conflict_exp

    Returns
    -------

    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    times = pd.DataFrame()
    relax = pd.DataFrame()

    queries = 950 if conflict_exp else 1000
    rand = RandomMLCL()
    constraints = rand.fit(pd.DataFrame(dataset), oracle=MLCLOracle(budget=queries, truth=labels))
    l_time = []
    l_relax = []
    if conflict_exp:
        create_conflicts(constraints, 50)

    start = start_partition(args, battery, dataset_name)

    if algo == "IAC":
        # Interactive clustering loop
        framework = IAC(pd.DataFrame(dataset))
        framework.init_loop(start)
        framework.constraints = [constraints]
        # framework.select_constraints(constraints) ne marche pas sur cette expé, très bizarre
        delta = 1 if conflict_exp else 0.94
        _, t = framework.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=0.3, sat_rate=delta))
        l_time.append(t)
        nb_unsat, rate = satisfaction_rate(framework.partition, constraints)
        l_relax.append(nb_unsat)
        framework.get_partitions(f"{args.o}/relaxing/{dataset_name}/raw/partitions{i + 1}_{algo}_{conflict_exp}")
        framework.get_constraints(f"{args.o}/relaxing/{dataset_name}/raw/constraints{i + 1}_{algo}_{conflict_exp}")
    else:
        alg = getattr(asc, algo)
        partitions = {0: start}
        l_constraints = []
        l_constraints.append(constraints)
        try:
            t = time()
            cop = alg(n_clusters=len(set(partitions[0])))
            cop.fit(dataset, ml=constraints["ml"], cl=constraints["cl"])
            l_time.append(time() - t)
            partitions[1] = cop.labels_
            nb_unsat, rate = satisfaction_rate(cop.labels_, constraints)
            l_relax.append(nb_unsat)
        except Exception as e:
            partitions[1] = partitions[0]
            l_time.append(np.NaN)
            l_relax.append(np.NaN)
        pd.DataFrame(partitions).to_csv(f"{args.o}/relaxing/{dataset_name}/raw/partitions{i + 1}_{algo}_{conflict_exp}.csv")
        get_constraints(l_constraints, f"{args.o}/relaxing/{dataset_name}/raw/constraints{i + 1}_{algo}_{conflict_exp}")

    relax[i + 1] = l_relax
    relax.to_csv(f"{args.o}/relaxing/{dataset_name}/raw/relax{i + 1}_{algo}_{conflict_exp}.csv")
    times[i + 1] = l_time
    times.to_csv(f"{args.o}/relaxing/{dataset_name}/raw/times{i + 1}_{algo}_{conflict_exp}.csv")


def create_conflicts(constraints, n):
    ct_list = constraints["ml"] + constraints["cl"]
    conflicts = []
    for i in range(n):
        x, y = random.choice(ct_list)
        if (x, y) in constraints["ml"]:
            constraints["cl"].append((y, x))
        elif (x, y) in constraints["cl"]:
            constraints["ml"].append((x, y))
        conflicts.append((x, y))
    return conflicts
