import numpy as np
import pandas as pd
import clustbench
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing
import os
from func_timeout import func_timeout, FunctionTimedOut


def load_dataset(args, name, battery=None):
    """
    Convenience function for loading a dataset.
    Parameters
    ----------
    args
    name
    battery

    Returns
    -------

    """
    if battery:
        loader = clustbench.load_dataset(battery, name, path=args.path)
        labels = loader.labels[0] - 1  # correspondence between clustbench and Python indexing
        n_clusters = loader.n_clusters[0]
        dataset = loader.data
    else:
        dataset = pd.read_csv(f"datasets/{name}/dataset.csv", header=None).to_numpy()
        labels = pd.read_csv(f"datasets/{name}/ground_truth.csv", header=None).T.to_numpy()[0]
        n_clusters = len(np.unique(labels))
    return pd.DataFrame(dataset), labels, n_clusters


def start_partition(args, battery, name):
    """
    Convenience function for getting a start partition for the incremental process.

    Parameters
    ----------
    args
    battery
    name

    Returns
    -------

    """
    if not os.path.exists(f"{args.o}/starts/{name}.csv"):
        dataset, labels, K = load_dataset(args, name, battery)
        start_partition = KMeans(n_init="auto", random_state=9, n_clusters=K)
        start_partition.fit(dataset)
        pd.DataFrame(start_partition.labels_).to_csv(f"{args.o}/starts/{name}.csv", index=False, header=False)
        return start_partition.labels_
    else:
        return pd.read_csv(f"{args.o}/starts/{name}.csv", header=None).T.to_numpy()[0]


def num_ground_truth(gt):
    num_gt = np.zeros(len(gt))
    cls = np.unique(gt)
    for i in range(len(gt)):
        num_gt[i] = np.where(cls == gt[i])[0]
    return num_gt


def exp_loop(args, experiments, directory, params1, params2, n_runs, f):
    """
    Experimental loop iterating over two lists of parameter values.

    Parameters
    ----------
    args
    experiments
    directory
    params1
    params2
    n_runs
    f

    Returns
    -------

    """
    for battery in experiments:
        for dataset_name in experiments[battery]:
            print(f"============================== Dataset : {dataset_name} ==============================")
            if not os.path.exists(f"{args.o}/{directory}/{dataset_name}"):
                os.mkdir(f"{args.o}/{directory}/{dataset_name}")
                os.mkdir(f"{args.o}/{directory}/{dataset_name}/raw")
                os.mkdir(f"{args.o}/{directory}/{dataset_name}/compiled")

            for alpha in params1:
                for beta in params2:
                    suffix = f"{alpha}_{beta}"
                    if os.path.exists(f"{args.o}/{directory}/{dataset_name}/compiled/compiled_{suffix}.csv"):
                        print(f"{suffix} : experiment already done")
                        continue

                    jobs = 15 if dataset_name == "digits" else 90
                    try:
                        print(f"+++++++++++++++++++++++++++ Config : {suffix} +++++++++++++++++++++++++++")
                        # for i in range(n_runs):
                        #    func_timeout(5000, f, args=(args, battery, dataset_name, i, t, n))
                        Parallel(n_jobs=1, verbose=10, timeout=5000)(delayed(f)(args, battery, dataset_name, i, alpha, beta) for i in range(n_runs))
                        print("Compiling...", sep="")
                        from experiments.results import compile_results
                        compile_results(args, directory, suffix, battery, dataset_name, n_runs)
                        print("Done")
                    except TimeoutError or multiprocessing.context.TimeoutError or FunctionTimedOut:
                        print(f"Config : {alpha}, {beta} did not finish before timeout")
                        continue
