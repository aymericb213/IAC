import pandas as pd
from iac import IAC
from modification import UACM, MWCM
from skquery.pairwise import NPU
from skquery.oracle import MLCLOracle
from experiments.exp_utils import load_dataset, start_partition


def sensitivity_analysis(args, battery, dataset_name, i, o, s):
    """
    Make an experimental run on a dataset with set parameters for IAC.

    Parameters
    ----------
    args : object
        Parsed input arguments.
    battery : str
        Name of the dataset battery.
    dataset_name : str
        Name of the dataset to test.
    i : int
        ID of the experimental run.
    o : float
        Value of the objective rate of IAC.
    s : float
        Value of the generalization rate of IAC.

    Returns
    -------
    Results are written in text files according to the type of measure.
    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    time = pd.DataFrame()
    times = []
    start = start_partition(args, battery, dataset_name)

    framework = IAC(pd.DataFrame(dataset))
    framework.init_loop(start)
    n = None
    while framework.ask_for_termination(args.iter, auto=args.auto) != "y":
        active = NPU(n)
        framework.select_constraints(active, MLCLOracle(budget=10, truth=labels))
        _, t = framework.modify_partition(UACM(MWCM, objective_rate=o, generalization_rate=s))
        times.append(t)
        n = active.neighborhoods

    framework.get_partitions(f"{args.o}/params/{dataset_name}/raw/partitions{i + 1}_{o}_{s}")
    framework.get_constraints(f"{args.o}/params/{dataset_name}/raw/constraints{i + 1}_{o}_{s}")
    time[i + 1] = times
    time.to_csv(f"{args.o}/params/{dataset_name}/raw/times{i + 1}_{o}_{s}.csv")