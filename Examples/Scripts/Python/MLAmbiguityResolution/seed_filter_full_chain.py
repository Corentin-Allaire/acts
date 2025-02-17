import glob

import pandas as pd
import numpy as np

import torch.utils
import argparse

from sklearn.cluster import DBSCAN

from train_seed_solver import prepareTrainingData, result_optuna
from seed_solver_network import prepareDataSet

def readDataSet(CKS_files: list[str]) -> pd.DataFrame:
    """Read the dataset from the different files, remove the pure duplicate tracks and combine the datasets"""
    """
    @param[in] CKS_files: DataFrame contain the data from each track files (1 file per events usually)
    @return: combined DataFrame containing all the track, ordered by events and then by truth particle ID in each events
    """
    data = []
    for f in CKS_files:
        datafile = pd.read_csv(f)
        datafile = prepareDataSet(datafile)
        data.append(datafile)
    return data

def clusterSeed(
    event: pd.DataFrame, DBSCAN_eps: float = 0.1, DBSCAN_min_samples: int = 2, weight: list[float] = [1, 1, 50, 1]
) -> pd.DataFrame:
    """
    Cluster together all the track that appear to belong to the same truth particle
    To cluster the tracks together, a DBSCAN is first used followed by a sub clustering based on hits shared by tracks.
    """
    """
    @param[in] event: input DataFrame that contain all track in one event
    @param[in] DBSCAN_eps: minimum radius used by the DBSCAN to cluster track together
    @param[in] DBSCAN_min_samples: minimum number of tracks needed for DBSCAN to create a cluster
    @return: DataFrame identical to the output with an added column with the cluster
    """
    # Perform the DBSCAN clustering and sort the Db by cluster ID
    trackDir = event[["eta", "phi", "vertexZ", "pT"]].to_numpy()
    if len(weight) != len(trackDir[0]):
        raise RuntimeError("Weight lenght", len(weight), "differ from input lenght", len(trackDir)[0]) 
    for i in range(len(weight)):
        trackDir[:, i] = trackDir[:, i] / weight[i]
    # Perform the subclustering
    clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(trackDir)
    clusterarray = renameCluster(clustering.labels_)
    event["cluster"] = clusterarray
    sorted = event.sort_values(["cluster"], ascending=True)
    return sorted


def renameCluster(clusterarray: np.ndarray) -> np.ndarray:
    """Rename the cluster IDs to be int starting from 0"""
    """
    @param[in] clusterarray: numpy array containing the hits IDs and the cluster ID
    @return: numpy array with updated cluster IDs
    """
    new_id = len(set(clusterarray)) - (1 if -1 in clusterarray else 0)
    for i, cluster in enumerate(clusterarray):
        if cluster == -1:
            clusterarray[i] = new_id
            new_id = new_id + 1
    return clusterarray


def cleanData(data, threshold, DBSCAN_eps: float = 0.1, DBSCAN_min_samples: int = 2, weight: list[float] = [1, 1, 50, 1], random = False):

    sc = "score"
    if random:
        sc = "random_score"
    cleanedData = []
    for event in data:
        if not len(event) == 0:
            cleanedEvent = clusterSeed(event, DBSCAN_eps, DBSCAN_min_samples, weight)

            # For each cluster only keep the seed with the highest score
            idmax = (
                cleanedEvent.groupby(["cluster"])[sc].transform("max")
                == cleanedEvent[sc]
            )
            cleanedEvent = cleanedEvent[idmax]
            # For cluster with more than 1 seed, keep the one with the smallest seed_id
            idfirst = (
                cleanedEvent.groupby(["cluster"])["seed_id"].transform("min")
                == cleanedEvent["seed_id"]
            )
            cleanedEvent = cleanedEvent[idfirst]

        idx = cleanedEvent[sc] > threshold
        cleanedEvent = cleanedEvent[idx]
        cleanedData.append(cleanedEvent)
    return cleanedData


def evalPerf(data, cleanedData, time, silent = False):
    nb_part = 0
    nb_track = 0
    nb_fake = 0
    nb_duplicate = 0

    nb_good_match = 0
    nb_reco_part = 0
    nb_reco_fake = 0
    nb_reco_duplicate = 0
    nb_reco_track = 0

    for event, cleanedEvent in zip(data, cleanedData):
        nb_part += event.loc[
            event["good/duplicate/fake"] != "fake"
        ].index.nunique()
        nb_track += event.shape[0]
        nb_fake += event.loc[
            event["good/duplicate/fake"] == "fake"
        ].shape[0]
        nb_duplicate += event.loc[
            event["good/duplicate/fake"] == "duplicate"
        ].shape[0]

        nb_good_match += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "good"
        ].shape[0]
        nb_reco_fake += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "fake"
        ].shape[0]
        nb_reco_duplicate += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "duplicate"
        ].shape[0]
        nb_reco_part += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] != "fake"
        ].index.nunique()
        nb_reco_track += cleanedEvent.shape[0]

    if not silent:

        print("===Initial efficiencies===")
        print("nb events", len(cleanedData))
        print("nb particles: ", nb_part)
        print("nb track: ", nb_track)
        print("duplicate rate: ", 100 * nb_duplicate / nb_track, " %")
        print("Fake rate: ", 100 * nb_fake / nb_track, " %")

        print("===computed efficiencies===")
        print("nb particles: ", nb_part)
        print("nb good match: ", nb_good_match)
        print("nb particle reco: ", nb_reco_part)
        print("nb track reco: ", nb_reco_track)
        print("Efficiency (good track): ", 100 * nb_good_match / nb_part, " %")
        print("Efficiency (particle reco): ", 100 * nb_reco_part / nb_part, " %")
        print(
            "duplicate rate: ",
            100 * ((nb_good_match + nb_reco_duplicate) - nb_reco_part) / nb_reco_track,
            " %",
        )
        print("Fake rate: ", 100 * nb_reco_fake / nb_reco_track, " %")

        print("===computed speed===")
        print("Load: ", (time[1] - time[0]) * 1000 / len(CKF_files), "ms")
        print("Inference: ", (time[2] - time[1]) * 1000 / len(CKF_files), "ms")
        print("Clustering: ", (time[3] - time[2]) * 1000 / len(CKF_files), "ms")

        print("tot: ", (time[3] - time[0]) * 1000 / len(CKF_files), "ms")
        print("Seed filter: ", (time[3] - time[1]) * 1000 / len(CKF_files), "ms")
    return (100 * nb_reco_part / nb_part) + (10 * nb_good_match / nb_part) - 20*((nb_track- nb_part)/nb_track)

def perfPloting(data, cleanedData):

    import matplotlib.pyplot as plt

    # Make a copy of the data to be plotted
    plotDF = pd.DataFrame()
    plotDF_duplicate = pd.DataFrame()
    plotDF_cluster = pd.DataFrame()
    plotDF_good_cluster = pd.DataFrame()
    plotDF_good_missing = pd.DataFrame()  # Initialize before loop

    for event, cleanEvent in zip(data, cleanedData):
        # Extract the duplicate seed and look at their maximum distance per particle in eta, phi, vertexZ and pT.
        # This will be use the to tune the DBScan reweighting (the width of the 4 distribution after rewighting should be similar)
        duplicate = event.loc[event["good/duplicate/fake"] == "duplicate"].copy()
        weight = [0.1, 1, 10, 20]
        duplicate.loc[:, "diff_eta"] = (duplicate.groupby(["particleId"])["eta"].transform("max") - duplicate.groupby(["particleId"])["eta"].transform("min")) / weight[0]
        duplicate.loc[:, "diff_phi"] = (duplicate.groupby(["particleId"])["phi"].transform("max") - duplicate.groupby(["particleId"])["phi"].transform("min")) / weight[1]
        duplicate.loc[:, "diff_vertexZ"] = (duplicate.groupby(["particleId"])["vertexZ"].transform("max") - duplicate.groupby(["particleId"])["vertexZ"].transform("min")) / weight[2]
        duplicate.loc[:, "diff_pT"] = (duplicate.groupby(["particleId"])["pT"].transform("max") - duplicate.groupby(["particleId"])["pT"].transform("min")) / weight[3]
        plotDF = pd.concat([plotDF, event.copy()])
        plotDF_duplicate = pd.concat([plotDF_duplicate, duplicate])

        # Compute number of seed per cluter and per type of seed
        seed_info = event.copy()
        seed_info["particleId"] = seed_info.index
        seed_info["nb_seed"] = 0
        seed_info["nb_fake"] = 0
        seed_info["nb_duplicate"] = 0
        seed_info["nb_good"] = 0
        seed_info["nb_truth"] = 0
        seed_info["nb_seed_truth"] = 0
        seed_info["nb_seed_removed"] = 0

        seed_info["nb_seed"] = seed_info.groupby(["cluster"])["cluster"].transform("size")
        # Create histogram filled with the number of fake seeds per cluster
        seed_info["nb_fake"] = seed_info.groupby("cluster")["good/duplicate/fake"].transform(
            lambda x: (x == "fake").sum()
        )
        # Create histogram filled with the number of duplicate seeds per cluster
        seed_info["nb_duplicate"] = seed_info.groupby("cluster")["good/duplicate/fake"].transform(
            lambda x: (x == "duplicate").sum()
        )
        # Create histogram filled with the number of good seeds per cluster
        seed_info["nb_good"] = seed_info.groupby("cluster")["good/duplicate/fake"].transform(
            lambda x: (x == "good").sum()
        )
        # Create histogram filled with the number unique particle per cluster
        seed_info["nb_truth"] = seed_info.groupby(["cluster"])["particleId"].transform('nunique')

        # Filter the cluster associated with good particles
        good_clusters = seed_info[event["good/duplicate/fake"] == "good"]["cluster"].unique()
        filtered_good = seed_info[event["cluster"].isin(good_clusters)]

        plotDF_cluster = pd.concat([plotDF_cluster, seed_info])
        plotDF_good_cluster = pd.concat([plotDF_good_cluster, filtered_good])


        fake_only_particleIds = cleanEvent.groupby("particleId").filter(
            lambda x: x["good/duplicate/fake"].nunique() == 1 and x["good/duplicate/fake"].iloc[0] == "fake"
        ).index.unique()
        
        # Add particles that are in event but not in cleanEvent
        missing_particleIds = event.loc[~event.index.isin(cleanEvent.index)].index.unique()
        fake_only_particleIds = np.concatenate((fake_only_particleIds, missing_particleIds))


        good_missing = event.loc[event["good/duplicate/fake"] == "good"]
        good_missing = good_missing[good_missing.index.isin(fake_only_particleIds)]

        plotDF_good_missing = pd.concat([plotDF_good_missing, good_missing])



    # Plot the distribution of the 4 variable positional seed variable
    plotDF["eta"].hist(bins=100)
    plt.xlabel("eta")
    plt.ylabel("nb seed")
    plt.savefig("eta.png")
    plt.clf()

    plotDF["phi"].hist(bins=100)
    plt.xlabel("phi")
    plt.ylabel("nb seed")
    plt.savefig("phi.png")
    plt.clf()

    plotDF["vertexZ"].hist(bins=100)
    plt.xlabel("vertexZ")
    plt.ylabel("nb seed")
    plt.savefig("vertexZ.png")
    plt.clf()

    plotDF["pT"].hist(bins=100, range=[0, 10])
    plt.xlabel("pT")
    plt.ylabel("nb seed")
    plt.savefig("pT.png")
    plt.clf()

    plotDF.plot.scatter(x="eta", y="pT")
    plt.xlabel("eta")
    plt.ylabel("pT")
    plt.savefig("pT_eta.png")
    plt.clf()

    # Plot the maximum distance between duplicated seed in the 4 DBScan variables
    plotDF_duplicate["diff_eta"].hist(bins=100, range=[0, 1])
    plt.xlabel("diff_eta")
    plt.ylabel("nb particle")
    plt.savefig("diff_eta.png")
    plt.clf()

    plotDF_duplicate["diff_phi"].hist(bins=100, range=[0, 1])
    plt.xlabel("diff_phi")
    plt.ylabel("nb particle")
    plt.savefig("diff_phi.png")
    plt.clf()

    plotDF_duplicate["diff_vertexZ"].hist(bins=100, range=[0, 1])
    plt.xlabel("diff_vertexZ")
    plt.ylabel("nb particle")
    plt.savefig("diff_vertexZ.png")
    plt.clf()

    plotDF_duplicate["diff_pT"].hist(bins=100, range=[0, 1])
    plt.xlabel("diff_pT")
    plt.ylabel("nb particle")
    plt.savefig("diff_pT.png")
    plt.clf()

    # Plot number of seed per cluster
    plotDF_cluster["nb_seed"].hist(bins=21, weights=1 / plotDF_cluster["nb_seed"], range=[-0.5, 20.5])
    plt.xlabel("nb seed/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_seed.png")
    plt.clf()

    plotDF_cluster["nb_fake"].hist(bins=21, weights=1 / plotDF_cluster["nb_seed"], range=[-0.5, 20.5])
    plt.xlabel("nb fake/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_fake.png")
    plt.clf()

    plotDF_cluster["nb_duplicate"].hist(bins=21, weights=1 / plotDF_cluster["nb_seed"], range=[-0.5, 20.5])
    plt.xlabel("nb duplicate/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_duplicate.png")
    plt.clf()

    plotDF_cluster["nb_good"].hist(bins=5, weights=1 / plotDF_cluster["nb_seed"], range=[-0.5, 4.5])
    plt.xlabel("nb good/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_good.png")
    plt.clf()

    plotDF_cluster["nb_truth"].hist(bins=5, weights=1 / plotDF_cluster["nb_seed"], range=[-0.5, 4.5])
    plt.xlabel("nb truth particle/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_particle.png")
    plt.clf()


    # Plot number of seed per good cluster (at leas 1 good seed)
    plotDF_good_cluster["nb_seed"].hist(bins=20, weights=1 / plotDF_good_cluster["nb_seed"], range=[0, 20])
    plt.xlabel("nb seed/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_seed.png")
    plt.clf()

    plotDF_good_cluster["nb_good"].hist(bins=5, weights=1 / plotDF_good_cluster["nb_seed"], range=[-0.5, 4.5])
    plt.xlabel("nb good/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("good_cluster_nb_good_.png")
    plt.clf()

    plotDF_good_cluster["nb_fake"].hist(bins=10, weights=1 / plotDF_good_cluster["nb_seed"], range=[-0.5, 9.5])
    plt.xlabel("nb fake/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("good_cluster_nb_fake.png")
    plt.clf()

    plotDF_good_cluster["nb_duplicate"].hist(bins=10, weights=1 / plotDF_good_cluster["nb_seed"], range=[-0.5, 9.5])
    plt.xlabel("nb duplicate/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("good_cluster_nb_duplicate.png")
    plt.clf()

    plotDF_good_cluster["nb_truth"].hist(bins=5, weights=1 / plotDF_good_cluster["nb_seed"], range=[-0.5, 4.5])
    plt.xlabel("nb truth particle/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("good_cluster_nb_particle.png")
    plt.clf()


    # Plot the distribution for truth particle that are removed by the filter
    plotDF_good_missing["score"].hist(bins=100, range=[0, 1])
    plt.xlabel("score")
    plt.ylabel("number of particle")
    plt.title("Score distribution for removed particles")
    plt.savefig("innefficiency_score.png")
    plt.clf()

    plotDF_good_missing["eta"].hist(bins=100, range=[-3, 3])
    plt.xlabel("eta")
    plt.ylabel("number of particle")
    plt.title("eta distribution for removed particles")
    plt.savefig("innefficiency_eta.png")
    plt.clf()

    plotDF_good_missing["phi"].hist(bins=100, range=[-3, 3])
    plt.xlabel("phi")
    plt.ylabel("number of particle")
    plt.title("phi distribution for removed particles")
    plt.savefig("innefficiency_phi.png")
    plt.clf()

    plotDF_good_missing["pT"].hist(bins=40, range=[-0.5, 39.5])
    plt.xlabel("score")
    plt.ylabel("number of particle")
    plt.title("pT distribution for removed particles")
    plt.savefig("innefficiency_pT.png")
    plt.clf()

    # Plot the score and variable distribution for the 3 different type of seed
    plt.figure()
    for tag in ["good", "duplicate", "fake"]:
        weights = np.ones_like(
            plotDF.loc[plotDF["good/duplicate/fake"] == tag]["score"]
        ) / len(
            plotDF.loc[plotDF["good/duplicate/fake"] == tag]["score"]
        )
        plt.hist(
            plotDF.loc[plotDF["good/duplicate/fake"] == tag]["score"],
            bins=100,
            weights=weights,
            alpha=0.65,
            label=tag,
        )
    plt.legend()
    plt.xlabel("score")
    plt.ylabel("Fraction of good/duplicate/fake tracks")
    plt.title("Score distribution for each type of track")
    plt.savefig("score_distribution.png")
    plt.yscale("log")
    plt.savefig("score_distribution_log.png")

    # Average value of the score
    averageDataPlots = plotDF.loc[
        plotDF["good/duplicate/fake"] == "good"
    ].groupby(
        pd.cut(
            plotDF.loc[plotDF["good/duplicate/fake"] == "good"]["eta"],
            np.linspace(-3, 3, 100),
        ),
        observed=False,
    )
    plt.figure()
    plt.plot(
        np.linspace(-3, 3, 99),
        averageDataPlots["score"].mean(),
        label="average score",
    )
    plt.legend()
    plt.xlabel("eta")
    plt.ylabel("score")
    plt.title("Average score for each eta bin")
    plt.savefig("score_eta.png")

    # Plot the pT distribution for each type of track
    plt.figure()
    plt.hist(
        [
            plotDF.loc[plotDF["good/duplicate/fake"] == "good"][
                "pT"
            ],
            plotDF.loc[
                plotDF["good/duplicate/fake"] == "duplicate"
            ]["pT"],
            plotDF.loc[plotDF["good/duplicate/fake"] == "fake"][
                "pT"
            ],
        ],
        bins=100,
        range=(0, 100),
        stacked=False,
        label=["good", "duplicate", "fake"],
    )
    plt.legend()
    plt.xlabel("pT")
    plt.ylabel("number of tracks")
    plt.yscale("log")
    plt.title("pT distribution for each type of track")
    plt.savefig("distribution_pT.png")

    # Plot the eta distribution for each type of track
    plt.figure()
    plt.hist(
        [
            plotDF.loc[plotDF["good/duplicate/fake"] == "good"][
                "eta"
            ],
            plotDF.loc[
                plotDF["good/duplicate/fake"] == "duplicate"
            ]["eta"],
            plotDF.loc[plotDF["good/duplicate/fake"] == "fake"][
                "eta"
            ],
        ],
        bins=100,
        range=(-3, 3),
        stacked=False,
        label=["good", "duplicate", "fake"],
    )
    plt.legend()
    plt.xlabel("eta")
    plt.ylabel("number of tracks")
    plt.yscale("log")
    plt.title("eta distribution for each type of track")
    plt.savefig("distribution_eta.png")

    # Average value of the score for 50 pt bins
    averageDataPlots = plotDF.loc[
        plotDF["good/duplicate/fake"] == "good"
    ].groupby(
        pd.cut(
            plotDF.loc[plotDF["good/duplicate/fake"] == "good"]["pT"],
            np.linspace(0, 100, 50),
        ), 
        observed=False,
    )
    plt.figure()
    plt.plot(
        np.linspace(0, 100, 49),
        averageDataPlots["score"].mean(),
        label="average score",
    )
    plt.legend()
    plt.xlabel("pT")
    plt.ylabel("score")
    plt.title("Average score for each eta bin")
    plt.savefig("score_pt.png")
    plt.clf()

def result_optuna(study):
    """Print the result of the optuna hyperparameter tuning and save the plots"""
    """
    @param[in] study: optuna study containing the result of the hyperparameter tuning
    """

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    # To be updated, write information on the best trial
    # Some additional plotting will need to be added
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optimization_history.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image("parallel_coordinate.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("param_importances.png")

    fig = optuna.visualization.plot_slice(study)
    fig.write_image("slice.png")

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image("intermediate_values.png")

    fig = optuna.visualization.plot_contour(study)
    fig.write_image("contour.png")

# ==================================================================

if __name__ == "__main__":
    import time

    timing = []

    timing.append(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",help="Input files for the test",default="/workdir/allairec/ruche_track_seed/event20[0-9][0-9]-seed_cleaned.csv")
    parser.add_argument("--model",help="Input files for the test",default="/gpfs/workdir/allairec/SolverNetwork/train_seed_optimise_lay3/seed_solver_optimisation_best.pt")
    parser.add_argument("--optimise", help="Turn on optuna hyperparameter tuning", action='store_true', default=False)
    args = parser.parse_args()
    
    # ttbar events as test input
    CKF_files = sorted(glob.glob(args.input))
    data = readDataSet(CKF_files)

    duplicateClassifier = torch.load(args.model, map_location=torch.device('cpu'), weights_only=False)

    # Data of each event after clustering
    # Data of each event after ambiguity resolution

    timing.append(time.time())

    # Performed the MLP based ambiguity resolution
    for event in data:
        # Prepare the data
        x_test, y_test = prepareTrainingData(event)
        x = torch.tensor(x_test, dtype=torch.float32)
        output_predict = duplicateClassifier(x).detach().numpy()

        event["score"] = output_predict
        event["random_score"] = np.random.rand(len(x_test))
        event["cluster"] = 0


    # Optimise mode, optuna will be used to try to find the ideal network size and parameters
    if args.optimise:
        try:
            # If in optimise mode, try to import optuna
            import kaleido
            import plotly
            import optuna
        except ImportError:
            print("Optuna (or kaleido and plotly) is not installed, please install it using 'pip install optuna'")
            exit()

        # Create an objective function that will be optimised later on
        def objective(trial):
            data_copy = data.copy()
            threshold = trial.suggest_float("threshold", 0.05, 1)
            DBSCAN_eps = trial.suggest_float("DBSCAN_eps", 0.01, 3)
            DBSCAN_min_samples = trial.suggest_int("DBSCAN_min_samples", 1, 10)
            weight = [0.1, 1, 10, 20]
            timing = [0,1,2,3]
            cleanedData = cleanData(data_copy, threshold, DBSCAN_eps, DBSCAN_min_samples, weight)
            score = evalPerf(data_copy, cleanedData, timing, True)
            del data_copy, cleanedData
            return score

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
               min_resource=1, max_resource=10, reduction_factor=3
            ),
        )

        study.optimize(objective, n_trials=1000)

        result_optuna(study)

        best_trial = study.best_trial
        best_params = best_trial.params
        timing.append(time.time())
        cleanedData = cleanData(data, best_params["threshold"], best_params["DBSCAN_eps"],  best_params["DBSCAN_min_samples"], [0.1, 1, 10, 20])
        timing.append(time.time())
        evalPerf(data, cleanedData, timing)
        perfPloting(data, cleanedData)

    else:

        timing.append(time.time())
        cleanedData = cleanData(data, 0.3, 0.1, 2, [0.1, 1, 10, 20])
        timing.append(time.time())
        evalPerf(data, cleanedData, timing)
        perfPloting(data, cleanedData)




