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
        idx = event[sc] > threshold
        cleanedEvent = event[idx].copy()
        if not len(cleanedEvent) == 0:
            cleanedEvent = clusterSeed(cleanedEvent, DBSCAN_eps, DBSCAN_min_samples, weight)

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
    return (100 * nb_reco_part / nb_part) + (10 * nb_good_match / nb_part) - ((nb_track- nb_part)/nb_track)

def perfPloting(data, cleanedData):

    import matplotlib.pyplot as plt

    # Make a copy of the data to be plotted
    plotData = []
    plotDF = pd.DataFrame()
    for event in data:
        plotData.append(event.copy())
        plotDF = pd.concat([plotDF, event.copy()])

    # Plot the distribution of the 4 variable
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


    plotDF2 = pd.DataFrame()
    # Create histogram filled with the number of seeds per cluster
    for event in cleanedData:
        event["nb_seed"] = 0
        event["nb_fake"] = 0
        event["nb_duplicate"] = 0
        event["nb_good"] = 0
        event["nb_cluster"] = 0
        event["nb_truth"] = 0
        event["nb_seed_truth"] = 0
        event["nb_seed_removed"] = 0
        event["particleId"] = event.index
        event["nb_seed"] = event.groupby(["cluster"])["cluster"].transform("size")
        # Create histogram filled with the number of fake seeds per cluster
        event.loc[event["good/duplicate/fake"] == "fake", "nb_fake"] = (
            event.loc[event["good/duplicate/fake"] == "fake"]
            .groupby(["cluster"])["cluster"]
            .transform("size")
        )
        # Create histogram filled with the number of duplicate seeds per cluster
        event.loc[event["good/duplicate/fake"] == "duplicate", "nb_duplicate"] = (
            event.loc[event["good/duplicate/fake"] == "duplicate"]
            .groupby(["cluster"])["cluster"]
            .transform("size")
        )
        # Create histogram filled with the number of good seeds per cluster
        event.loc[event["good/duplicate/fake"] == "good", "nb_good"] = (
            event.loc[event["good/duplicate/fake"] == "good"]
            .groupby(["cluster"])["cluster"]
            .transform("size")
        )

        plotDF2 = pd.concat([plotDF2, event])

    plotDF2["nb_seed"].hist(bins=20, weights=1 / plotDF2["nb_seed"], range=[0, 20])
    plt.xlabel("nb seed/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_seed.png")
    plt.clf()

    plotDF2["nb_fake"].hist(bins=10, weights=1 / plotDF2["nb_seed"], range=[0, 10])
    plt.xlabel("nb fake/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_fake.png")
    plt.clf()

    plotDF2["nb_duplicate"].hist(bins=10, weights=1 / plotDF2["nb_seed"], range=[0, 10])
    plt.xlabel("nb duplicate/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_duplicate.png")
    plt.clf()

    plotDF2["nb_good"].hist(bins=5, weights=1 / plotDF2["nb_seed"], range=[0, 5])
    plt.xlabel("nb good/[cluster]")
    plt.ylabel("Arbitrary unit")
    plt.savefig("nb_good.png")
    plt.clf()

    # ==================================================================
    # Plotting

    # Combine the events to have a better statistic
    DataPlots = pd.concat(data)

    import matplotlib.pyplot as plt

    # Plot the average score distribution for each type of track
    plt.figure()
    for tag in ["good", "duplicate", "fake"]:
        weights = np.ones_like(
            DataPlots.loc[DataPlots["good/duplicate/fake"] == tag]["score"]
        ) / len(
            DataPlots.loc[DataPlots["good/duplicate/fake"] == tag]["score"]
        )
        plt.hist(
            DataPlots.loc[DataPlots["good/duplicate/fake"] == tag]["score"],
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
    averageDataPlots = DataPlots.loc[
        DataPlots["good/duplicate/fake"] == "good"
    ].groupby(
        pd.cut(
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "good"]["eta"],
            np.linspace(-3, 3, 100),
        )
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
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "good"][
                "pT"
            ],
            DataPlots.loc[
                DataPlots["good/duplicate/fake"] == "duplicate"
            ]["pT"],
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "fake"][
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
    plt.savefig("pT_distribution.png")

    # Plot the eta distribution for each type of track
    plt.figure()
    plt.hist(
        [
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "good"][
                "eta"
            ],
            DataPlots.loc[
                DataPlots["good/duplicate/fake"] == "duplicate"
            ]["eta"],
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "fake"][
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
    plt.savefig("eta_distribution.png")

    # Average value of the score for 50 pt bins
    averageDataPlots = DataPlots.loc[
        DataPlots["good/duplicate/fake"] == "good"
    ].groupby(
        pd.cut(
            DataPlots.loc[DataPlots["good/duplicate/fake"] == "good"]["pT"],
            np.linspace(0, 100, 50),
        )
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
    parser.add_argument("--model",help="Input files for the test",default="/gpfs/workdir/allairec/SolverNetwork/train_seed_2layer/trained_seed_solver_best.pt")
    parser.add_argument("--optimise", help="Turn on optuna hyperparameter tuning", action='store_true', default=False)
    args = parser.parse_args()
    
    # ttbar events as test input
    CKF_files = sorted(glob.glob(args.input))
    data = readDataSet(CKF_files)

    duplicateClassifier = torch.load(args.model, map_location=torch.device('cpu'))

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
            threshold = trial.suggest_float("threshold", 0.0, 1)
            DBSCAN_eps = trial.suggest_float("DBSCAN_eps", 0.001, 5)
            DBSCAN_min_samples = trial.suggest_int("DBSCAN_min_samples", 1, 10)
            weight_eta = trial.suggest_float("weight_eta", 1, 5)
            weight_phi = trial.suggest_float("weight_phi", 1, 5)
            weight_vertexZ = trial.suggest_float("weight_vertexZ", 1, 100)
            weight_pT = trial.suggest_float("weight_pT", 1, 100)
            weight = [weight_eta, weight_phi, weight_vertexZ, weight_pT]
            timing = [0,1,2,3]
            cleanedData = cleanData(data_copy, threshold, DBSCAN_eps, DBSCAN_min_samples, weight)
            score = evalPerf(data_copy, cleanedData, timing, False)
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
        cleanedData = cleanData(data, best_params["threshold"], best_params["DBSCAN_eps"],  best_params["DBSCAN_min_samples"],  [best_params["weight_eta"], best_params["weight_phi"], best_params["weight_vertexZ"], best_params["weight_pT"]])
        timing.append(time.time())
        evalPerf(data, cleanedData, timing)
        perfPloting(data, cleanedData)

    else:

        timing.append(time.time())
        cleanedData = cleanData(data, 0.2, 0.2, 2, [1, 1, 50, 1])
        timing.append(time.time())
        evalPerf(data, cleanedData, timing)
        perfPloting(data, cleanedData)




