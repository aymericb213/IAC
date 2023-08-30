import os
import argparse
from experiments import *
from selection.model_matching import IMM
from experiments.exp_utils import load_dataset
import plotly.express as px


def menu(args, choice=0, n_runs=0):
    if not os.path.exists(args.o):
        os.mkdir(args.o)
        os.mkdir(f"{args.o}/plots")
        os.mkdir(f"{args.o}/starts")

    experiments = {"other": ["iris"],
                   "fcps": ["lsun", "chainlink", "target", "atom", "engytime", "wingnut"],
                   "uci": ["wine", "glass", "yeast", "sonar", "ionosphere", "statlog", "ecoli"],
                   None: ["letters"], "mnist": ["digits"]}
    args.iter = 10
    dir_name, params1, params2, exp_function = "", [], [], None

    choice = int(input("Available experiments:\n"
                       "1: sensitivity analysis (Section 4.2.1)\n"
                       "2: COP scaling (Section 4.2.2)\n"
                       "3: constraint relaxation (Section 4.2.3)\n"
                       "4: constraint cumulativeness"
                       "5: comparison in incremental setting (Section 4.2.4)\n"
                       "6: use case on SITS (Section 4.3)\n"
                       "Choose an experiment by typing its number: ")) if choice == 0 else choice
    match choice:
        case 1:
            dir_name = "params"
            params1 = [0, 0.05, 0.2]
            params2 = [0.1, 0.3, 0.5, 1]
            exp_function = sensitivity_analysis
        case 2:
            experiments = {"uci": ["yeast"], None: ["letters"], "mnist": ["digits"]}
            args.iter = 1
            dir_name = "scaling"
            params1 = ["mlcl", "triplet", "span-specific", "span-generic", "mlcl+triplet", "all"]
            params2 = [10, 100, 1000]
            exp_function = scaling_experiment
        case 3:
            args.iter = 1
            experiments = {"wut": ["mk2"]}
            dir_name = "relaxing"
            params1 = ["IAC", "PCKMeans", "MPCKMeans"]
            params2 = [False, True]
            exp_function = relax_experiment
        case 4:
            dir_name = "cumulativeness"
            params1 = ["RandomMLCL"]
            params2 = ["once", "incr", "cumul"]
            exp_function = cumulativeness
        case 5:
            dir_name = "comparison"
            params1 = ["IAC", "COPKMeans", "PCKMeans", "MPCKMeans"]
            params2 = ["Random", "NPUincr"]
            exp_function = comparison
        case 6:
            if not os.path.exists(f"{args.o}/treecut/"):
                os.mkdir(f"{args.o}/treecut/")
            print("Running use case experiment")
            tc_use_case(args)
            print(f"Experiment finished, results available in {args.o}/treecut")
            return
        case _:
            print("Input not recognized (answer 1 to 6). Going back to menu.")
            menu(args)
    if not os.path.exists(f"{args.o}/{dir_name}/"):
        os.mkdir(f"{args.o}/{dir_name}/")
        os.mkdir(f"{args.o}/plots/{dir_name}/")
        os.mkdir(f"{args.o}/plots/{dir_name}/qual")
        os.mkdir(f"{args.o}/plots/{dir_name}/sim")
        os.mkdir(f"{args.o}/plots/{dir_name}/time")
        os.mkdir(f"{args.o}/plots/{dir_name}/boxplots")
    n_runs = int(input("Choose number of runs (90 is used in the paper) : ")) if n_runs == 0 else n_runs
    print(f"Running {exp_function.__name__} experiment ({n_runs} runs)")
    exp_loop(args, experiments, dir_name, params1, params2, n_runs, exp_function)
    dirs = [d for d in next(os.walk(f"{args.o}/{dir_name}"))[1]]
    for dataset in dirs:
        plot_aubc(args, dataset, dir_name, params1, params2, n_runs)
        #boxplot_aubc(args, dataset, dir_name, params1, params2, n_runs)
        #plot_time(args, dataset, dir_name, params1, params2, n_runs)
    #barplot_time(args, dir_name, params1, params2, n_runs)
    #bayesian_validation(args, dirs, dir_name, n_runs, params1, params2)
    #compute_wins(args, dir_name)
    print(f"Experiment finished, results available in {args.o}/{dir_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IAC experimental setup')
    parser.add_argument('-path', type=str, default="clustering-data-v1-1.1.0", help='path to clustbench')
    parser.add_argument("-iter", type=int, default=10, help='number of iterations')
    parser.add_argument('--auto', action=argparse.BooleanOptionalAction, default=True, help='auto mode')
    parser.add_argument('-o', type=str, default="tests", help='output path')

    args = parser.parse_args()
    args.iter = 1
    print("Experiments from CP'23 paper 38 'Incremental constrained clustering by minimal weighted modification'")
    menu(args, 1, 10)

