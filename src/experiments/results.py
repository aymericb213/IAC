import baycomp
import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product, combinations
import plotly.graph_objects as go
import plotly.express as px
from experiments.exp_utils import load_dataset

METRICS = ["ARI", "AMI", "FMI"]
TYPES = ["qual", "sim"]


def confidence_interval(mean, std, sample_size, conf_rate, interval=True):
    assert sample_size > 30
    match conf_rate:
        case 0.9:
            z = 1.645
        case 0.95:
            z = 1.960
        case 0.99:
            z = 2.576
        case _:
            raise ValueError("Non-standard confidence rate given")
    return z * (std / np.sqrt(sample_size)) if not interval else (mean - z * (std / np.sqrt(sample_size)), mean + z * (std / np.sqrt(sample_size)))


def compile_results(args, directory, suffix, battery, dataset_name, n_runs):
    """
    Processes results of all experimental runs, computes raw and average metrics
    written in CSV files.

    Parameters
    ----------
    args : Namespace
        Parsed input arguments.
    directory : str
        Directory in which to write the results.
    suffix : str
    battery
    dataset_name
    n_runs

    Returns
    -------

    """
    dataset, labels, K = load_dataset(args, dataset_name, battery)

    # pair_dist = pairwise_distances(dataset, n_jobs=-1) if len(labels) < 25000 else pairwise_distances(dataset[:1000], n_jobs=-1)
    # sample = None if len(labels) < 25000 else 1000
    raw_results_qual = {}
    raw_results_sim = {}
    raw_time = {}
    raw_relax = {}
    for i in range(n_runs):
        df = pd.read_csv(f"{args.o}/{directory}/{dataset_name}/raw/partitions{i + 1}_{suffix}.csv", index_col=0)
        raw_results_qual[f"ARI {i}"] = [adjusted_rand_score(df[f"{j}"].T.to_numpy(), labels) for j in range(df.shape[1])]
        raw_results_qual[f"AMI {i}"] = [adjusted_mutual_info_score(df[f"{j}"].T.to_numpy(), labels) for j in range(df.shape[1])]
        raw_results_qual[f"FMI {i}"] = [fowlkes_mallows_score(df[f"{j}"].T.to_numpy(), labels) for j in range(df.shape[1])]
        # raw_results_qual[f"Silhouette {i}"] = [silhouette_score(pair_dist, df.loc[:999, f"{j}"].T.to_numpy(), sample_size=sample) for j in range(args.iter+1)]
        raw_results_sim[f"ARI {i}"] = [adjusted_rand_score(df[f"{j + 1}"].T.to_numpy(), df[f"{j}"].T.to_numpy()) for j in range(df.shape[1] - 1)]
        raw_results_sim[f"AMI {i}"] = [adjusted_mutual_info_score(df[f"{j + 1}"].T.to_numpy(), df[f"{j}"].T.to_numpy()) for j in range(df.shape[1] - 1)]
        raw_results_sim[f"FMI {i}"] = [fowlkes_mallows_score(df[f"{j + 1}"].T.to_numpy(), df[f"{j}"].T.to_numpy()) for j in range(df.shape[1] - 1)]

        time = pd.read_csv(f"{args.o}/{directory}/{dataset_name}/raw/times{i + 1}_{suffix}.csv", index_col=0)
        raw_time[i] = time.iloc[:, 0]
        if directory == "relaxing":
            relax = pd.read_csv(f"{args.o}/{directory}/{dataset_name}/raw/relax{i + 1}_{suffix}.csv", index_col=0)
            raw_relax[i] = relax.iloc[:, 0]

    raw_results_qual = pd.DataFrame(raw_results_qual)
    raw_results_sim = pd.DataFrame(raw_results_sim)
    raw_time = pd.DataFrame(raw_time)

    raw_results_qual.to_csv(f"{args.o}/{directory}/{dataset_name}/compiled/raw_qual_{suffix}.csv")
    raw_results_sim.to_csv(f"{args.o}/{directory}/{dataset_name}/compiled/raw_sim_{suffix}.csv")
    raw_time.to_csv(f"{args.o}/{directory}/{dataset_name}/compiled/raw_time_{suffix}.csv")
    if directory == "relaxing":
        raw_relax = pd.DataFrame(raw_relax)
        raw_relax.to_csv(f"{args.o}/{directory}/{dataset_name}/compiled/raw_relax_{suffix}.csv")

    averaged_results = {}
    for metric in METRICS:
        averaged_results[f"qual mean {metric}"] = [np.mean([raw_results_qual.loc[j, f"{metric} {i}"] for i in range(n_runs)]) for j in range(raw_results_qual.shape[0])]
        averaged_results[f"qual std {metric}"] = [np.std([raw_results_qual.loc[j, f"{metric} {i}"] for i in range(n_runs)]) for j in range(raw_results_qual.shape[0])]
        if metric != "Silhouette":
            # for similarity, first row is sum of differences over iterations
            averaged_results[f"sim mean {metric}"] = [np.mean([sum([1 - raw_results_sim.loc[i, f"{metric} {j}"] for i in range(raw_results_sim.shape[0])]) for j in range(n_runs)])] + [
                np.mean([raw_results_sim.loc[j, f"{metric} {i}"] for i in range(n_runs)]) for j in range(raw_results_sim.shape[0])]
            averaged_results[f"sim std {metric}"] = [np.std([sum([1 - raw_results_sim.loc[i, f"{metric} {j}"] for i in range(raw_results_sim.shape[0])]) for j in range(n_runs)])] + [
                np.std([raw_results_sim.loc[j, f"{metric} {i}"] for i in range(n_runs)]) for j in range(raw_results_sim.shape[0])]
    # for time, first row is total time
    averaged_results["time mean"] = [np.mean([sum([raw_time.iloc[i, j] for i in range(raw_time.shape[0])]) for j in range(n_runs)])] + [np.mean(raw_time.iloc[i, :]) for i in range(raw_time.shape[0])]
    averaged_results["time std"] = [np.std([sum([raw_time.iloc[i, j] for i in range(raw_time.shape[0])]) for j in range(n_runs)])] + [np.std(raw_time.iloc[i, :]) for i in range(raw_time.shape[0])]
    if directory == "relaxing":
        averaged_results["# relaxed"] = [np.mean([sum([raw_relax.iloc[i, j] for i in range(raw_relax.shape[0])]) for j in range(n_runs)])] + [np.mean(raw_relax.iloc[i, :]) for i in
                                                                                                                                              range(raw_relax.shape[0])]
    pd.DataFrame(averaged_results).to_csv(f"{args.o}/{directory}/{dataset_name}/compiled/compiled_{suffix}.csv")


def compute_aubc(dataset, results, n_runs):
    x_qual = [i * 0.1 for i in range(len(results[0]))]
    x_sim = [i * 0.1 for i in range(len(results[1]))]
    metrics = {}
    match len(results):
        case 2:  # raw
            for metric in METRICS:
                aubcs_q = [auc(x_qual, [results[0].loc[j, f"{metric} {i}"] for j in range(len(results[0]))]) for i in range(n_runs)]
                if len(x_sim) > 1:
                    aubcs_s = [auc(x_sim, [results[1].loc[j, f"{metric} {i}"] for j in range(len(results[1]))]) for i in range(n_runs)]
                else:
                    aubcs_s = np.zeros(n_runs)
                metrics[metric] = (aubcs_q, aubcs_s)
            return metrics
        case _:  # averaged
            for metric in METRICS:
                metrics[metric] = []
                for kind in TYPES:
                    values = results[f"{kind} mean {metric}"].T.to_numpy()
                    stds = results[f"{kind} std {metric}"].T.to_numpy()
                    ci_bounds = [confidence_interval(values[i], stds[i], n_runs, 0.95) for i in range(len(values))]
                    if kind == "qual":
                        lb = auc(x_qual, [x[0] for x in ci_bounds])
                        aubc = auc(x_qual, [values[i] for i in range(len(values))])
                        ub = auc(x_qual, [x[1] for x in ci_bounds])
                    else:
                        lb = auc(x_sim, MinMaxScaler().fit_transform(np.reshape([x[0] for x in ci_bounds], (-1, 1))))
                        aubc = auc(x_sim, MinMaxScaler().fit_transform(np.reshape([values[i] for i in range(1, len(values))], (-1, 1))))
                        ub = auc(x_sim, MinMaxScaler().fit_transform(np.reshape([x[1] for x in ci_bounds], (-1, 1))))
                    metrics[metric].append((lb, aubc, ub))
            return metrics


def plot_aubc(args, dataset, directory, params1, params2, n_runs):
    methods = list(product(params1, params2))
    for metric in METRICS:
        kinds = TYPES if metric != "Silhouette" else ["qual"]
        for kind in kinds:
            fig = go.Figure()
            if "IAC" in params1:
                names = iter([f"{x[0]}+{x[1]}" for x in list(product(["IAC", "COP", "PCK", "MPCK"], ["Rand", "NPU"]))])
            elif directory == "cumulativeness":
                names = iter(["once", "incr", "cumul"])
            else:
                names = iter([f"IAC({x[0]},{x[1]})" for x in list(product([0, 0.05, 0.2], [0.1, 0.3, 0.5, 1]))])
            markers = iter(["cross", "diamond", "circle", "hexagon", "square", "triangle-up", "pentagon", "triangle-down", "star", "bowtie", "octagon", "hourglass"])
            min = []
            for method in methods:
                x = [i * 10 for i in range(11)] if kind == "qual" else [i * 10 for i in range(1, 11)]
                results = pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/compiled_{method[0]}_{method[1]}.csv", index_col=0)
                values = results[f"{kind} mean {metric}"].T.to_numpy()
                stds = results[f"{kind} std {metric}"].T.to_numpy()
                if "once" in method:
                    x = [0, 100] if kind == "qual" else [100]
                min.append(np.min(values))
                bounds = [confidence_interval(values[i], stds[i], n_runs, 0.95, False) for i in range(len(values))]
                y = values if kind == "qual" else values[1:]
                fig.add_trace(go.Scatter(x=x,  # values
                                         y=y,
                                         mode='lines+markers',
                                         name=next(names),
                                         error_y=go.scatter.ErrorY(array=bounds),
                                         marker=go.scatter.Marker(symbol=next(markers), size=10)
                                         ))
            fig.update_layout(template="simple_white",
                              legend={"x": 0.05, "y": 0.95, "font": {"size": 20}, "borderwidth": 1},
                              legend_title="Method",
                              font={"size": 25},
                              # showlegend=False
                              )
            fig.update_xaxes(title="Number of queries", range=[0, 105], dtick=10)
            fig.update_yaxes(title=f"{metric}")
            if kind == "sim":
                fig.update_yaxes(range=[np.min(min) - 0.05, 1])
            fig.write_html(f"{args.o}/plots/{directory}/{dataset}_{metric}_{kind}.html")


def barplot_time(args, directory, params1, params2, n_runs):
    experiments = ["iris", "wine", "sonar", "glass", "ecoli", "ionosphere", "lsun", "target", "atom", "chainlink", "wingnut", "yeast", "statlog", "engytime", "letters", "digits"]

    methods = list(product(params1, params2))
    fig = go.Figure(layout=go.Layout(xaxis_title="Dataset", yaxis_title="CPU time (s)", template="simple_white",
                                     legend={"orientation": "h",
                                             "x": 1, "y": 1,
                                             "yanchor": "bottom",
                                             "xanchor": "right", },
                                     legend_title="Methods",
                                     font={"size": 26}
                                     ))
    colors = iter(["blue", "lightblue", "red", "tomato", "darkgreen", "limegreen", "purple", "orchid"])
    if directory == "comparison":
        names = iter([f"{x[0]}+{x[1]}" for x in list(product(["IAC", "COP", "PCK", "MPCK"], ["Rand", "NPU"]))])
    else:
        names = iter(["once", "incr", "cumul"])
    for method in methods:
        y = []
        err_y = []
        for dataset in experiments:
            try:
                results = pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/compiled_{method[0]}_{method[1]}.csv", index_col=0)
                values = results[f"time mean"].T.to_numpy()
                bound = confidence_interval(np.mean(values[1:]), np.std(values[1:]), n_runs, 0.95, False)
                y.append(np.mean(values[1:]))
                err_y.append(bound)
            except FileNotFoundError:
                y.append(np.NaN)
                err_y.append(np.NaN)
        fig.add_trace(go.Bar(name=next(names),
                             x=experiments, y=y,
                             error_y=go.bar.ErrorY(array=err_y),
                             ))
        print(method)
        fig['data'][-1]['marker']['color'] = next(colors)
        fig['data'][-1]['marker']['pattern'] = go.bar.marker.Pattern(shape="/") if "NPU" in method[1] else go.bar.marker.Pattern(shape="")

    fig.write_html(f"{args.o}/plots/{directory}/time/time.html")
    fig.update_yaxes(type="log", showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.write_html(f"{args.o}/plots/{directory}/time/time_log.html")


def plot_time(args, dataset, directory, cst_type, n_constraints, n_runs):
    fig = go.Figure()
    colors = iter(["red", "green", "blue", "orange", "purple", "gray"])
    ci_colors = iter(["rgba(255,0,0,0.2)", "rgba(0,255,0,0.2)", "rgba(0,0,255,0.2)", "rgba(255,200,0,0.2)", "rgba(255,0,255,0.2)", "rgba(89,55,29,0.2)"])
    markers = iter(["circle", "square", "triangle-down", "triangle-up", "cross", "star"])

    for kind in cst_type:
        y = []
        color = next(colors)
        ci_color = next(ci_colors)
        marker = next(markers)
        for n in n_constraints:
            try:
                results = pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/compiled_{kind}_{n}.csv", index_col=0)
                value = results.loc[0, "time mean"]
                std = results.loc[0, "time std"]
                lb, ub = confidence_interval(value, std, n_runs, 0.95)
                y.append((lb, value, ub))
            except FileNotFoundError:
                y.append((np.NaN, np.NaN, np.NaN))
        print(y)
        fig.add_trace(go.Scatter(x=n_constraints,  # values
                                 y=[x[1] for x in y],
                                 mode='lines+markers',
                                 name=kind,
                                 marker=go.scatter.Marker(symbol=marker, size=17),
                                 line_color=color
                                 ))
        fig.add_traces([go.Scatter(x=n_constraints,  # lower bound
                                   y=[x[0] for x in y],
                                   mode='lines', line_color=ci_color,
                                   name='95% CI (lower)',
                                   showlegend=False),
                        go.Scatter(x=n_constraints,  # upper bound
                                   y=[x[2] for x in y],
                                   mode='lines', line_color=ci_color,
                                   name='95% CI (upper)',
                                   fill='tonexty', fillcolor=ci_color,
                                   showlegend=False)])
    fig.update_layout(  # title=f"{dataset}",
        template="simple_white",
        legend={"x": 0.05, "y": 0.95, "borderwidth": 1},
        legend_title="Type of constraint",
        font={"size": 25}
    )
    fig.update_xaxes(title="Number of constraints", type='category')
    fig.update_yaxes(title="CPU time (s)", type="log", showgrid=True, gridwidth=1, gridcolor='LightGray', tickfont={"size": 22})
    fig.write_html(f"{args.o}/plots/{directory}/plot_{dataset}.html")
    fig.write_image(f"{args.o}/plots/{directory}/plot_{dataset}.svg")


def compile_aubc(args, experiments, directory, n_runs, params1, params2):
    for n in params1:
        for t in params2:
            results = {"Dataset": [], "ARI qual lb": [], "ARI qual": [], "ARI qual ub": [],
                       "ARI sim lb": [], "ARI sim": [], "ARI sim ub": [],
                       "AMI qual lb": [], "AMI qual": [], "AMI qual ub": [],
                       "AMI sim lb": [], "AMI sim": [], "AMI sim ub": [],
                       "FMI qual lb": [], "FMI qual": [], "FMI qual ub": [],
                       "FMI sim lb": [], "FMI sim": [], "FMI sim ub": []}
            for dataset in experiments:
                metrics = compute_aubc(dataset, pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/compiled_{n}_{t}.csv", index_col=0), n_runs)
                results["Dataset"] += [dataset]
                for metric in metrics:
                    results[f"{metric} qual lb"] += [metrics[metric][0][0]]
                    results[f"{metric} qual"] += [metrics[metric][0][1]]
                    results[f"{metric} qual ub"] += [metrics[metric][0][2]]
                    results[f"{metric} sim lb"] += [metrics[metric][1][0]]
                    results[f"{metric} sim"] += [metrics[metric][1][1]]
                    results[f"{metric} sim ub"] += [metrics[metric][1][2]]
            pd.DataFrame(results).to_csv(f"{args.o}/{directory}/AUBC_{n}_{t}.csv", index=False)


def bayesian_validation(args, experiments, directory, n_runs, params1, params2):
    methods = list(product(params1, params2))
    for metric in METRICS:
        aubcs_q = []
        aubcs_s = []
        # Computing likelihood functions
        for (n, t) in methods:
            matrix_q, matrix_s = np.zeros((len(experiments), n_runs)), np.zeros((len(experiments), n_runs))
            for i in range(len(experiments)):
                results = compute_aubc(experiments[i], (pd.read_csv(f"{args.o}/{directory}/{experiments[i]}/compiled/raw_qual_{n}_{t}.csv", index_col=0),
                                                        pd.read_csv(f"{args.o}/{directory}/{experiments[i]}/compiled/raw_sim_{n}_{t}.csv", index_col=0)),
                                       n_runs)
                matrix_q[i] = results[metric][0]
                matrix_s[i] = results[metric][1]
            aubcs_q.append(matrix_q)
            aubcs_s.append(matrix_s)
        print(aubcs_q[0].shape)
        # Bayesian comparison
        for kind in TYPES:
            if os.path.exists(f"{args.o}/{directory}/bayesian_comparison_{metric}_{kind}.csv"):
                print("Comparisons already made")
                pass
            else:
                aubcs = aubcs_q if kind == "qual" else aubcs_s
                results = pd.DataFrame(columns=["conf 1", "conf 2", "P_left", "P_rope", "P_right"])
                tests, params = list(combinations(aubcs, 2)), list(combinations(methods, 2))
                for i in range(len(tests)):
                    method1, method2 = tests[i]
                    while True:
                        try:
                            probs, fig = baycomp.two_on_multiple(method1, method2, rope=0.01, plot=True, nsamples=50000,
                                                                 names=(f"{params[i][0][1]}",
                                                                        f"{params[i][1][1]}"))
                            break
                        except Exception as e:  # relaunch when httpstan server timeouts
                            print(e)
                            print(f"Exception encountered, relaunching stan...")
                            pass
                    fig.savefig(f"{args.o}/plots/{directory}/{kind}/comp_{params[i][0]}-{params[i][1]}_{metric}_{kind}.svg")
                    results = pd.concat([results, pd.Series([params[i][0], params[i][1], probs[0], probs[1], probs[2]], index=results.columns).to_frame().T])
                results.to_csv(f"{args.o}/{directory}/bayesian_comparison_{metric}_{kind}.csv", index=False)


def compute_wins(args, directory, alpha=0.95):
    wins = {}
    for metric in METRICS:
        for kind in TYPES:
            counter = {}
            df = pd.read_csv(f"{args.o}/{directory}/bayesian_comparison_{metric}_{kind}.csv")
            for i in range(df.shape[0]):
                conf1, conf2, p1, p_rope, p2 = df.iloc[i, :]
                if conf1 not in counter:
                    counter[conf1] = 0
                if conf2 not in counter:
                    counter[conf2] = 0
                if alpha > 0:
                    if p1 > alpha:
                        counter[conf1] += 1
                    elif p2 > alpha:
                        counter[conf2] += 1
                else:
                    if p1 > p2 + p_rope:
                        counter[conf1] += 1
                    elif p2 > p1 + p_rope:
                        counter[conf2] += 1
            wins[f"{metric}_{kind}"] = counter
    print(wins)
    return wins


def pairwise_plot(args):
    def improve_text_position(x):
        # fix indentation
        positions = ['top center']  # you can add more: left center ...
        return [positions[i % len(positions)] for i in range(len(x))]

    x = [[9, 2, 0, 0, 10, 7, 1, 5, 10, 8, 4, 5], [10, 1, 0, 4, 10, 5, 2, 4, 10, 6, 3, 7], [9, 1, 0, 3, 10, 7, 1, 5, 10, 7, 4, 5]]
    y = [[0, 3, 6, 9, 1, 4, 6, 9, 1, 4, 6, 9], [0, 2, 6, 9, 1, 4, 6, 9, 2, 4, 6, 9], [0, 1, 4, 9, 1, 3, 4, 9, 1, 3, 6, 9]]
    for metric in range(3):
        qual = x[metric]
        sim = y[metric]
        index = ["(0,10)", "(0,30)", "(0,50)", "(0,100)",
                 "(5,10)", "(5,30)", "(5,50)", "(5,100)",
                 "(20,10)", "(20,30)", "(20,50)", "(20,100)", ]
        pareto = ["(20,10)", "(20,30)", "(20,100)"]
        fig = px.scatter(x=qual, y=sim, text=[x if x in pareto else "" for x in index], template="simple_white",
                         size=[20 if x in pareto else 5 for x in index],
                         color=["red" if x in pareto else "blue" for x in index])
        fig.update_traces(textposition="bottom center")
        fig.update_xaxes(title="Wins on quality")
        fig.update_yaxes(title="Wins on similarity")
        fig.update_layout(font={"size": 20}, showlegend=False)
        fig.show()
        fig.write_image(f"{args.o}/plots/prw_comp_{METRICS[metric]}.png")


def plot_partitions(args, directory, dataset, nb):
    partitions = pd.read_csv(f"{args.o}/{directory}/{dataset}/raw/partitions{nb}_IAC_NPUincr.csv", index_col=0)
    for i in range(partitions.shape[1]):
        fig = px.scatter(dataset, x=0, y=1, color=partitions.iloc[:, i],
                         color_continuous_scale=["#619FCA", "#FFA555", "#6ABC6A"],
                         template="simple_white")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()
        fig.write_image(f"lsun_partition{nb}_{i}.png")


def boxplot_aubc(args, dataset, directory, params1, params2, n_runs):
    methods = list(product(params1, params2))
    x = []
    y = []
    for (n, t) in methods:
        results = compute_aubc(None, (pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/raw_qual_{n}_{t}.csv", index_col=0),
                                      pd.read_csv(f"{args.o}/{directory}/{dataset}/compiled/raw_sim_{n}_{t}.csv", index_col=0)),
                               n_runs)
        y.append(results)
        x += [f"{n}_{t}" for _ in range(n_runs)]
    colors = ["blue"]*n_runs + ["lightblue"]*n_runs + ["red"]*n_runs + ["tomato"]*n_runs + ["darkgreen"]*n_runs + ["limegreen"]*n_runs + ["purple"]*n_runs + ["orchid"]*n_runs
    for metric in METRICS:
        quals = np.concatenate([method[metric][0] for method in y])
        sims = np.concatenate([method[metric][1] for method in y])
        df = pd.DataFrame({f"AUBC_quality({metric})": quals, f"AUBC_similarity({metric})": sims, "Method": x})
        fig = px.box(df, x="Method", y=f"AUBC_quality({metric})", title=dataset,
               template="simple_white", points="all")
        fig.update_layout(showlegend=False)
        fig.write_html(f"{args.o}/plots/{directory}/boxplots/boxplot_{dataset}_{metric}_qual.html")
        fig = px.box(df, x="Method", y=f"AUBC_similarity({metric})", title=dataset,
               template="simple_white", points="all")
        fig.update_layout(showlegend=False)
        fig.write_html(f"{args.o}/plots/{directory}/boxplots/boxplot_{dataset}_{metric}_sim.html")
