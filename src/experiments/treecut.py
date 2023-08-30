import pandas as pd
import numpy as np
import plotly.validators.layout.colorscale

from skimage import io
from matplotlib.colors import hsv_to_rgb
from modification import UACM, MWCM
from iac import IAC
import plotly.express as px
import plotly.graph_objects as go
from time import time
from copy import deepcopy

def treecut_data():
    constraints = []
    user_csts = pd.read_csv(f"use case/user_constraints.csv", header=None)
    for cst in user_csts.index:
        constraints.append(tuple(user_csts.loc[cst].to_numpy()))

    truth = pd.read_csv(f"use case/all_ground_truth.csv", header=None).T.to_numpy()[0]
    truth[truth == 1] = 0
    truth[truth == 2] = 1

    mat = pd.DataFrame(np.reshape(truth, (337, 724)))
    print(f"zone 1 : {mat.iloc[104:124, 192:207].sum().sum()} pixels")
    print(f"zone 2 : {mat.iloc[85:101, 282:299].sum().sum()} pixels")

    for mode in ["base"]:
        print(f"{mode} partition")
        init_partition = pd.read_csv(f"use case/init_partition_{mode}.csv", header=None).T.to_numpy()[0]

        clusters = (5, 14) if mode == "base" else (0, 11)

        cluster1 = np.copy(init_partition)  # 16188, 16203
        cluster1[init_partition != clusters[0]] = 0
        cluster1[init_partition == clusters[0]] = 1
        cluster2 = np.copy(init_partition)  # 53533, 53668
        cluster2[init_partition != clusters[1]] = 0
        cluster2[init_partition == clusters[1]] = 1
        pd.DataFrame(cluster1).to_csv(f"use case/cluster1_{mode}.csv", header=False, index=False)
        pd.DataFrame(cluster2).to_csv(f"use case/cluster2_{mode}.csv", header=False, index=False)
        cluster12 = np.copy(init_partition)
        cluster12[init_partition != clusters[0]] = 0
        cluster12[init_partition == clusters[0]] = 1
        cluster12[init_partition == clusters[1]] = 1
        pd.DataFrame(cluster12).to_csv(f"use case/cluster12_{mode}.csv", header=False, index=False)

        composition = {}
        constrained_points = set()
        for cst in constraints:
            constrained_points.add(cst[0])
            constrained_points.add(cst[1])
        constrained_points = sorted(list(constrained_points))

        for k in range(15):
            cuts = []
            not_cuts = []
            clust_ct_points = []
            for x in np.where(init_partition == k)[0]:
                lab = truth[x]
                if lab == 0:
                    not_cuts.append(x)
                else:
                    cuts.append(x)
                if x in constrained_points:
                    clust_ct_points.append(x)
            composition[k] = (cuts, not_cuts, clust_ct_points)
            #print(f"{k} : {len(cuts)} cuts, {len(cuts) + len(not_cuts)} total ({len(clust_ct_points)} constrained points)")


def treecut(args, init_fn):
    print("Composition of target tree cut zones")
    truth = pd.read_csv("use case/all_ground_truth.csv", header=None).T.to_numpy()[0]
    truth[truth == 1] = 0
    truth[truth == 2] = 1
    mat = pd.DataFrame(np.reshape(truth, (337, 724)))
    print(f"zone 1 : {mat.iloc[104:124, 192:207].sum().sum()} pixels")
    print(f"zone 2 : {mat.iloc[85:101, 282:299].sum().sum()} pixels")

    all_constraints = {"ml":[], "cl":[]}
    user_csts = pd.read_csv("use case/user_constraints.csv", header=None)
    for cst in user_csts.index:
        if user_csts.loc[cst, 2] == 1:
            all_constraints["ml"].append(tuple(user_csts.loc[cst,:1].to_numpy()))
        elif user_csts.loc[cst, 2] == -1:
            all_constraints["cl"].append(tuple(user_csts.loc[cst,:1].to_numpy()))

    print("Composition of initial partition in target")
    mat = pd.DataFrame(np.reshape(pd.read_csv(f"use case/{init_fn}.csv", header=None).T.to_numpy()[0], (337, 724)))
    print(f"zone 1 : {mat.iloc[104:124, 192:207].sum().sum()} pixels")
    print(f"zone 2 : {mat.iloc[85:101, 282:299].sum().sum()} pixels")
    f1 = None

    partition = pd.read_csv(f"use case/{init_fn}.csv", header=None).T.to_numpy()[0]
    prev_csts = {"ml":[], "cl":[]}
    print("Modifying partition")
    while True:
        constraints = {"ml":[], "cl":[]}
        for cst in all_constraints["ml"]:
            if partition[cst[0]] != partition[cst[1]]:
                constraints["ml"].append(cst)
        for cst in all_constraints["cl"]:
            if partition[cst[0]] == partition[cst[1]]:
                constraints["cl"].append(cst)
        if len(constraints["ml"]) + len(constraints["cl"]) == 1:
            break

        print(f"{sum([len(constraints[key]) for key in constraints])} constraints")
        t1 = time()
        clust = IAC(pd.read_csv("use case/tree_cut_data.csv"), truth)
        clust.init_loop(partition)
        clust.select_constraints(constraints)
        clust.modify_partition(UACM(MWCM, objective_rate=0.2, generalization_rate=1))

        print(f"Finished in {time() - t1} seconds")
        print("Composition of modified partition")
        mat = pd.DataFrame(np.reshape(clust.partition, (337, 724)))
        print(f"zone 1 : {mat.iloc[104:124, 192:207].sum().sum()} pixels")
        print(f"zone 2 : {mat.iloc[85:101, 282:299].sum().sum()} pixels")

        partition = np.copy(clust.partition)
        prev_csts = deepcopy(constraints)
        print(prev_csts)
        pd.DataFrame(partition).to_csv(f"{args.o}/treecut/final_full_partition.csv", header=False, index=False)
    return f1


def treecut_plot(args):
    # Ground truth
    truth = pd.read_csv("use case/all_ground_truth.csv", header=None)
    fig = px.imshow(np.reshape(truth.to_numpy(), (337, 724)))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    #fig.show()
    truth[truth == 1] = 0
    truth[truth == 2] = 1

    # Base image
    img = io.imread("use case/T32ULU_20170510T103031__sub_crop_NDVI.tif")
    #img = io.imread("./incremental_clustering_nour/data/Original_crop_images/T32ULU_20170510T103031__sub_crop.tif")
    fig = px.imshow(img, color_continuous_scale=plotly.colors.sequential.Viridis)
    fig.update_layout(coloraxis_showscale=False, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    #fig.show()

    constraints = []
    user_csts = pd.read_csv("use case/user_constraints.csv", header=None)
    for cst in user_csts.index:
        constraints.append(tuple(user_csts.loc[cst].to_numpy()))
    for cst in constraints:
        fig.add_trace(go.Scatter(name=str(cst), x=[cst[0] % 724, cst[1] % 724],
                                 mode="lines", y=[cst[0] // 724, cst[1] // 724]))
        if cst[-1] == 1:
            fig['data'][-1]['line']['color'] = "#ff0000"
        else:
            fig['data'][-1]['line']['color'] = "#0000ff"
            fig['data'][-1]['line']['dash'] = "dash"
    #fig.show()

    # Binary problem
    sep = pd.read_csv("use case/init_partition_base.csv", header=None)
    fig = px.imshow(np.reshape(sep.to_numpy(), (337, 724)))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    #fig.show()

    # Clusters for use case
    clustering = pd.read_csv("use case/cluster1_base.csv", header=None)
    case_cluster = clustering.loc[clustering[0] == 1].index.to_numpy()
    cluster_pixels = list(map(lambda i: (i % 724, i // 724), case_cluster))
    #print(len(cluster_pixels))

    # Final partition
    final = pd.read_csv(f"{args.o}/treecut/final_full_partition.csv", header=None)
    final_cluster = np.where(final == 1)[0]  # cluster where we want all tree cuts to belong
    final_pixels = list(map(lambda i: (i % 724, i // 724), final_cluster))
    #print(len(final_pixels))

    # Coupes de bois
    tree_cuts = np.where(truth == 1)[0]
    for pixels in [cluster_pixels, final_pixels]:
        true_pixels = list(map(lambda i: (i % 724, i // 724), tree_cuts))
        true_positives = set(true_pixels).intersection(pixels)
        false_negatives = set(true_pixels).difference(pixels)
        false_positives = set(pixels).difference(true_pixels)
        #print(len(true_positives))
        #print(len(false_positives))
        #print(len(false_negatives))

        # Image
        overlay = np.zeros((724, 337))
        layer2 = np.zeros((724, 337))
        layer3 = np.zeros((724, 337))
        layer4 = np.zeros((724, 337))

        for x, y in pixels:
            overlay[x, y] = 1
        for x, y in true_positives:
            layer2[x, y] = 1
        for x, y in false_negatives:
            layer3[x, y] = 1
        for x, y in false_positives:
            layer4[x, y] = 1

        hues = {0: np.reshape(truth.to_numpy(), (337, 724)), 280: overlay.T, 100: layer2.T, 50: layer3.T}
        name = {0: "GT", 280: "FP", 100: "TP", 50: "FN"}
        for x in hues:
            teinte = x  # (green=100, purple=280, red=0, yellow=50)

            hsv = np.ones((img.shape[0], img.shape[1], 3))
            hsv[:, :, 0] = teinte / 360
            hsv[:, :, 1] = hues[x]
            # image pure pour les zones non cluster, saturation max pour le cluster
            hsv[:, :, 2] = hues[x] + (1 - hues[x]) * (img - np.min(img)) / (np.max(img) - np.min(img))
            rgb = hsv_to_rgb(hsv)

            fig = px.imshow(rgb, color_continuous_scale='grey')
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            if pixels == cluster_pixels:
                fig.write_html(f"{args.o}/treecut/layer_{name[x]}_initial.html")
            if pixels == final_pixels:
                fig.write_html(f"{args.o}/treecut/layer_{name[x]}_final.html")
        # images have been assembled with Gimp

def tc_use_case(args):
    #treecut_data()
    treecut(args, "cluster1_base")
    treecut_plot(args)
