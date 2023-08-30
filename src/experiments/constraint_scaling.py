import pandas as pd
from iac import IAC
from modification import UACM, MWCM
from selection import RandomTriplet, RandomSpan
from skquery.pairwise import RandomMLCL
from skquery.oracle import MLCLOracle
from experiments.exp_utils import load_dataset, start_partition


def scaling_experiment(args, battery, dataset_name, i, cst_type, n_csts):
    """
    Make an experimental run on a dataset with a given type and number of constraints.

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
    cst_type : str
        Type of constraint to use, must be one of `mlcl`, `triplet`, `span-specific`,
        `span-generic`, `mlcl+triplet`, `all`.
    n_csts : int
        Number of constraints in the test set.

    Returns
    -------
    Results are written in text files according to the type of measure.
    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    times = pd.DataFrame()
    l_time = []
    start = start_partition(args, battery, dataset_name)
    # Interactive clustering loop
    framework = IAC(pd.DataFrame(dataset))
    framework.init_loop(start)
    while framework.ask_for_termination(args.iter, auto=args.auto) != "y":
        oracle = MLCLOracle(budget=n_csts, truth=labels)
        match cst_type:
            case "mlcl":
                active = RandomMLCL()
            case "triplet":
                active = RandomTriplet()
            case "span-specific":
                active = RandomSpan()
            case "span-generic":
                active = RandomSpan(generic=True)
            case "mlcl+triplet":
                active = combine_pairwise_triplet(framework.dataset, n_csts//2, labels)
                assert len(active["triplet"]) == n_csts//2
                assert len(active["ml"]) + len(active["cl"]) == n_csts//2
            case "all":
                active = combine_pairwise_triplet(framework.dataset, n_csts//2, labels)
                assert len(active["triplet"]) == n_csts//2
                assert len(active["ml"]) + len(active["cl"]) == n_csts//2
                sp = RandomSpan()
                oracle.budget = 1
                span_active = sp.fit(framework.dataset, oracle)
                active.update(span_active)
                sp_gen = RandomSpan(generic=True)
                span_gen_active = sp_gen.fit(framework.dataset, MLCLOracle(budget=1, truth=labels))
                active["span"].append(span_gen_active["span"][0])
                assert len(active["span"]) == 2
                print(active["span"])

        framework.select_constraints(active, oracle)
        print(framework.constraints[-1]["span"])
        t_solve, t = framework.modify_partition(UACM(MWCM, objective_rate=0, generalization_rate=1))
        l_time.append(t_solve)

    framework.get_partitions(f"{args.o}/scaling/{dataset_name}/raw/partitions{i + 1}_{cst_type}_{n_csts}")
    framework.get_constraints(f"{args.o}/scaling/{dataset_name}/raw/constraints{i + 1}_{cst_type}_{n_csts}")
    times[i + 1] = l_time
    times.to_csv(f"{args.o}/scaling/{dataset_name}/raw/times{i + 1}_{cst_type}_{n_csts}.csv")


def combine_pairwise_triplet(dataset, n_csts, labels):
    """
    Makes a set of random pairwise and triplet constraints.

    Parameters
    ----------
    dataset : matrix
        Dataset from which to select the constraints.
    n_csts : int
        Number of constraints to generate.
    labels : list of int
        Ground truth for the dataset.

    Returns
    -------
    Pairwise and triplet constraints.
    """
    tr = RandomTriplet()
    constraints = tr.fit(dataset,
                         MLCLOracle(budget=n_csts, truth=labels))
    triplet_pts = set()
    for tri in constraints["triplet"]:
        for x in tri:
            triplet_pts.add(x)
    mlcl_active = RandomMLCL()
    mlcl = mlcl_active.fit(dataset.iloc[list(triplet_pts), :],
                           MLCLOracle(budget=n_csts, truth=labels))
    constraints.update(mlcl)
    return constraints
