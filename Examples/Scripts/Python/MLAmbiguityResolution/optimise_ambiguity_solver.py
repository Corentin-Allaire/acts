import glob

import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

from ambiguity_solver_network import prepareDataSet, DuplicateClassifier, Normalise

import optuna
from optuna.trial import TrialState

avg_mean = [0, 0, 0, 0, 0, 0, 0, 0]
avg_sdv = [0, 0, 0, 0, 0, 0, 0, 0]
events = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def readDataSet(CKS_files: list[str]) -> pd.DataFrame:
    """Read the dataset from the different files, remove the pure duplicate tracks and combine the datasets"""
    """
    @param[in] CKS_files: DataFrame contain the data from each track files (1 file per events usually)
    @return: combined DataFrame containing all the track, ordered by events and then by truth particle ID in each events
    """
    data = pd.DataFrame()
    for f in CKS_files:
        datafile = pd.read_csv(f)
        # We at this point we don't make any difference between fake and duplicate
        datafile.loc[
            datafile["good/duplicate/fake"] == "fake", "good/duplicate/fake"
        ] = "duplicate"
        datafile = prepareDataSet(datafile)
        # Combine dataset
        data = pd.concat([data, datafile])
    return data


def prepareTrainingData(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the data"""
    """
    @param[in] data: input DataFrame to be prepared
    @return: array of the network input and the corresponding truth
    """
    # Remove truth and useless variable
    target_column = "rank"
    # Separate the truth from the input variables
    y = data[target_column]
    input = data.drop(
        columns=[
            target_column,
            "track_id",
            "seed_id",
            "nMajorityHits",
            "nSharedHits",
            "truthMatchProbability",
            "Hits_ID",
            "chi2",
            "pT",
            "good/duplicate/fake"  
        ]
    )
    # Compute the normalisation factors
    scale = StandardScaler()
    scale.fit(input.select_dtypes("number"))
    # Variables to compute the normalisation
    global avg_mean
    avg_mean = avg_mean + scale.mean_
    global avg_sdv
    avg_sdv = avg_sdv + scale.var_
    global events
    events = events + 1
    # Prepare the input feature
    x_cat = OrdinalEncoder().fit_transform(input.select_dtypes("object"))
    x = np.concatenate((x_cat, input), axis=1)
    return x, y


def batchSplit(data: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split the data into batch each containing @batch_size truth particles (the number of corresponding tracks may vary)"""
    """
    @param[in] data: input DataFrame to be cut into batch
    @param[in] batch_size: Number of truth particles per batch
    @return: list of DataFrame, each element correspond to a batch
    """
    batch = []
    pid = data[0][0]
    n_particle = 0
    id_prev = 0
    id = 0
    for index, row, truth in zip(data[0], data[1], data[2]):
        if index != pid:
            pid = index
            n_particle += 1
            if n_particle == batch_size:
                b = data[0][id_prev:id], data[1][id_prev:id], data[2][id_prev:id]
                batch.append(b)
                n_particle = 0
                id_prev = id
        id += 1
    return batch


def computeLoss(
    score_good: torch.Tensor,
    score_duplicate: list[torch.Tensor],
    batch_loss: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Compute one loss for each duplicate track associated with the particle"""
    """
    @param[in] score_good: score return by the model for the good track associated with this particle
    @param[in] score_duplicate: list of the scores of all duplicate track associated with this particle
    @param[in] margin: Margin used in the computation of the MarginRankingLoss
    @return: return the updated loss
    """
    # Compute the losses using the MarginRankingLoss with respect to the good track score
    batch_loss = batch_loss
    if score_duplicate:
        for i in range(len(score_duplicate)):
            batch_loss += F.relu(score_duplicate[i][1] - score_good + score_duplicate[i][0]*margin) / len(score_duplicate)
    return batch_loss


def scoringBatch(batch: list[pd.DataFrame], duplicateClassifier,  Optimiser=0) -> tuple[int, int, float]:
    """Run the MLP on a batch and compute the corresponding efficiency and loss. If an optimiser is specified train the MLP."""
    """
    @param[in] batch:  list of DataFrame, each element correspond to a batch
    @param[in] Optimiser: Optimiser for the MLP, if one is specify the network will be train on batch.
    @return: array containing the number of particles, the number of particle where the good track was found and the loss
    """
    # number of particles
    nb_part = 0
    # number of particles associated with a good track
    nb_good_match = 0
    # loss for the batch
    loss = 0
    # best track score for a particle
    max_score = 0
    # is the best score associated with the good track
    max_match = 0
    # loop over all the batch
    for b_data in batch:
        # ID of the current particule
        pid = b_data[0][0]
        # loss for the current batch
        batch_loss = 0
        # score of the good track
        score_good = 0
        # score of the duplicate tracks
        score_duplicate = []
        if Optimiser:
            Optimiser.zero_grad()
        input = torch.tensor(b_data[1], dtype=torch.float32)
        input = input.to(device)
        prediction = duplicateClassifier(input)
        # loop over all the track in the batch
        for index, pred, truth in zip(b_data[0], prediction, b_data[2]):
            # If we are changing particle uptade the loss
            if index != pid:
                # Starting a new particles, compute the loss for the previous one
                if max_match == 1:
                    nb_good_match += 1
                batch_loss = computeLoss(
                    score_good, score_duplicate, batch_loss, margin=0.015
                )
                nb_part += 1
                # Reinitialise the variable for the next particle
                pid = index
                score_duplicate = []
                score_good = 0
                max_score = 0
                max_match = 0
            # Store track scores
            if truth==0:
                score_good = pred
            else:
                score_duplicate.append([truth,pred])
            # Prepare efficiency computtion
            if pred == max_score:
                max_match = 0
            if pred > max_score:
                max_score = pred
                max_match = int(truth==0)
        # Compute the loss for the last particle when reaching the end of the batch
        if max_match == 1:
            nb_good_match += 1
        batch_loss = computeLoss(score_good, score_duplicate, batch_loss, margin=0.015)
        nb_part += 1
        # Normalise the loss to the batch size
        batch_loss = batch_loss / len(b_data[0])
        loss += batch_loss.item()
        # Perform the gradient descend if an optimiser was specified
        if Optimiser:
            batch_loss.backward()
            Optimiser.step()
    loss = loss / len(batch)
    return nb_part, nb_good_match, loss


def train(
    duplicateClassifier: DuplicateClassifier,
    trial: optuna.Trial,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    epoch: int = 0,
    batch: int = 32,
    validation: float = 0.3,
) -> DuplicateClassifier:
    """Training of the MLP"""
    """
    @param[in] duplicateClassifier: model to be trained.
    @param[in] data: tuple containing three list. Each element of those list correspond to a given track and represent : the truth particle ID, the track parameters and the truth.
    @param[in] epochs: number of epoch the model will be trained for.
    @param[in] batch: size of the batch used in the training
    @param[in] validation: Fraction of the batch used in training
    @return: trained model
    """
    # Prepare tensorboard for the training plot
    # use 'tensorboard --logdir=runs' to access the plot afterward
    opt = torch.optim.Adam(duplicateClassifier.parameters())
    # Split the data in batch
    batch = batchSplit(data, batch)
    val_batch = int(len(batch) * (1 - validation))
    print("Epoch: ", epoch, " / ", epochs)
    nb_part = 0.0
    nb_good_match = 0.0

    # Loop over all the network over the training batch
    nb_part, nb_good_match, _ = scoringBatch(batch[:val_batch], duplicateClassifier, Optimiser=opt)

    # If using validation, compute the efficiency and loss over the training batch
    if validation > 0.0:
        nb_part, nb_good_match, _ = scoringBatch(batch[val_batch:], duplicateClassifier)
        trial.report(nb_good_match / nb_part, epoch)

    return duplicateClassifier, nb_good_match / nb_part

# ==================================================================

# ttbar events used as the training input, here we assume 1000 events are availables
CKF_files = sorted(glob.glob("/workdir/allairec/ruche_track_ckf" + "/event[0-9000]-tracks_ckf.csv"))
#CKF_files = sorted(glob.glob("/workdir/allairec/ruche_track_ckf" + "/event[0-9]-tracks_ckf.csv"))
data = readDataSet(CKF_files)

epochs = 20
# Prepare the data
x_train, y_train = prepareTrainingData(data)

avg_mean = [x / events for x in avg_mean]
avg_sdv = [x / events for x in avg_sdv]

input = data.index, x_train, y_train

def objective(trial):

    # Create our model
    input_dim = np.shape(x_train)[1]
    layer_1 = trial.suggest_int("layer_1", 10, 100)
    layer_2 = trial.suggest_int("layer_2", 10, 100)
    layer_3 = trial.suggest_int("layer_3", 10, 100)
    layers_dim = [layer_1, layer_2, layer_3]

    duplicateClassifier = nn.Sequential(
        Normalise(avg_mean, avg_sdv), DuplicateClassifier(input_dim, layers_dim)
    )
    duplicateClassifier = duplicateClassifier.to(device)

    # Loop over the number of epochs
    for epoch in range(epochs):
        duplicateClassifier, score = train(
            duplicateClassifier, trial, input, epoch, batch=128, validation=0.3
        )

    return score

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=epochs, reduction_factor=3
    ),
)

study.optimize(objective, n_trials=120)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

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


