import pandas as pd
from iac import IAC
from modification import UACM, MWCM
from skquery.pairwise import RandomMLCL, NPU, AIPC
from skquery.oracle import MLCLOracle
from experiments.exp_utils import load_dataset, exp_loop, start_partition


def cumulativeness(args, battery, dataset_name, i, active, cumul):
    dataset, labels, K = load_dataset(args, dataset_name, battery)
    times = pd.DataFrame()
    l_time = []
    start = start_partition(args, battery, dataset_name)

    # Interactive clustering loop
    framework = IAC(pd.DataFrame(dataset))
    framework.init_loop(start)

    if cumul == "once":
        args.iter = 1
        selector = active()
        framework.select_constraints(selector, MLCLOracle(budget=100, truth=labels))
        _, t = framework.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=1))
        l_time.append(t)
    else:
        args.iter = 10
        n = None
        while framework.ask_for_termination(args.iter, auto=args.auto) != "y":
            selector = active(n) if active.__name__ == "NPU" else active()
            framework.select_constraints(selector, MLCLOracle(budget=10, truth=labels))
            if cumul == "incr":
                _, t = framework.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=0.3))
            else:
                ct_set = {"ml": [], "cl": []}
                for iteration in framework.constraints:
                    for key in ct_set:
                        ct_set[key] += iteration[key]
                framework.partition, mods, t = UACM(MWCM, objective_rate=0.2, generalization_rate=0.3).update(framework.dataset, framework.partition, framework.true_clusters, ct_set)
                framework.history.append(mods)
            l_time.append(t)
            n = selector.neighborhoods if active.__name__ == "NPU" else None
            if len(framework.constraints) > 1:
                print(f"Prior violated constraints : {framework.check_consistency()}")

    framework.get_partitions(f"{args.o}/cumulativeness/{dataset_name}/raw/partitions{i + 1}_{active.__name__}_{cumul}")
    framework.get_constraints(f"{args.o}/cumulativeness/{dataset_name}/raw/constraints{i + 1}_{active.__name__}_{cumul}")
    times[i+1] = l_time
    times.to_csv(f"{args.o}/cumulativeness/{dataset_name}/raw/times{i+1}_{active.__name__}_{cumul}.csv")