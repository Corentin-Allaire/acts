import glob

import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from ambiguity_solver_network import prepareDataSet, DuplicateClassifier, Normalise

import argparse

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


def batchSplit(data: tuple[torch.tensor, torch.tensor, torch.tensor], batch_size: int):
    """
    Split the data into batch each containing @batch_size truth particles (the number of corresponding seeds may vary)
    The batch is a list of tensor containing the boundaries from one particle ID to the next. 
    Each element of the list correspond to batch_size particles.
    """
    """
    @param[in] data: input to be cut into batch
    @param[in] batch_size: Number of truth particles per batch
    @return: A list of numpy.array, each array correspond to a batch. The index of a change of particle are store in the arrays
    """
    # Compute the boundaries between particle in the input dataset
    boundaries = torch.where(data[0][:-1] != data[0][1:])[0] + 1
    boundaries = torch.cat([torch.tensor([0], device=boundaries.device, requires_grad=False), boundaries, torch.tensor([len(data[0])], device=boundaries.device, requires_grad=False)])
    # Split the boundaries in batch of batch_size elements
    batch = torch.split(boundaries, batch_size)
    batch = [b.cpu().detach().numpy() for b in batch]

def scoringBatch(full_data, batch, duplicateClassifier, Optimiser=0):
    """Run the MLP on a batch and compute the corresponding efficiency and loss. If an optimiser is specify it also trains the MLP."""
    """
    @param[in] batch:  list of DataFrame, each element correspond to a batch 
    @param[in] Optimiser: Optimiser for the MLP, if one is specify the network will be train on batch. 
    @return: array containing the number of particles, the number of particle where the good seed was found and the loss
    """
    # number of particles
    nb_part = 0
    # number of particles associated with a good track
    nb_good_match = 0
    # loss for the batch
    loss = torch.tensor(0.0, device=device, requires_grad=False)

    # Extract the trainning part of the data (by oposition to the validation one)
    data = full_data[0][batch[0][0]:batch[-1][-1]], full_data[1][batch[0][0]:batch[-1][-1]], full_data[2][batch[0][0]:batch[-1][-1]]
    # Compute mask to select good, duplicate or fake
    good_mask = data[2] == 0
    duplicate_mask = data[2] > 0
    # Compute tensors that will be used to compute the loss as a single matrix multiplication
    rank = torch.clone(data[2])
    good = torch.zeros_like(data[2])
    good[duplicate_mask] = 1
    good[good_mask] = 1

    id_start = batch[0][0]
    # Loop over all the batch
    for i in range(len(batch)-1):

        if Optimiser:
            Optimiser.zero_grad()
        # Compute particle boundaries for the batch
        id_cut = batch[i][0]  - id_start
        id_next = batch[i+1][0] - id_start
        boundaries = batch[i] - id_cut - id_start
        boundaries = np.append(boundaries, id_next-id_cut)

        # Compute the MLP score
        predictions = duplicateClassifier(data[1][id_cut:id_next])
        predictions = torch.squeeze(predictions)
        # Extract the batch truth information
        truths_rank = rank[id_cut:id_next]

        good_score = predictions[good_mask[id_cut:id_next]]
        nb_part += len(good_score)
        nb_seed = torch.tensor([end-start for start, end in zip(boundaries[:-1], boundaries[1:])], device=device, requires_grad=False)
        good_score = good_score.repeat_interleave(nb_seed)
        # Compute a loss with 2 terms, a margin ranking loss to separate the good from duplicate 
        # and binary cross entropy to separate the fake from the rest
        batch_loss = torch.tensor(0.0, requires_grad=False, device=device)
        batch_loss += F.relu(predictions - good_score + truths_rank * duplicateClassifier[1].marginDuplicate).sum()

        # Normalize loss and accumulate
        batch_loss = batch_loss / len(predictions)
        loss += batch_loss
        # Perform gradient update if Optimiser is specified
        if Optimiser:
            batch_loss.backward()
            Optimiser.step()
            # used to ignore the efficiency computation in the training loop 
            nb_good_match = nb_part
        else:
            # Compute the efficiency in the validation pass
            # This is perform on CPU with numpy array as it is much faster
            pred = predictions.cpu().detach().numpy()
            g_mask = good_mask.cpu().detach().numpy()
            d_mask = duplicate_mask.cpu().detach().numpy()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                group_preds = pred[start:end]
                score_good =  group_preds[g_mask[id_cut+start:id_cut+end]]
                score_duplicate = group_preds[d_mask[id_cut+start:id_cut+end]]
                max_duplicate = score_duplicate.max() if score_duplicate.size > 0 else -1.0
                if score_good > max_duplicate:
                    nb_good_match += 1

    loss = loss.item() / len(batch)
    return nb_part, nb_good_match, loss


def train_epoch(
    duplicateClassifier: DuplicateClassifier,
    opt: torch.optim.Optimizer,
    data: tuple[torch.tensor, torch.tensor, torch.tensor],
    batch: list[np.ndarray],
    val_batch: int,
    epoch: int = 0,
    writer: SummaryWriter = None,
    trial=None,
) -> float :
    """Training one epoch of the model and compute the efficiency and loss over the training and validation batch"""
    """
    @param[in] duplicateClassifier: model to be trained.
    @param[in] data: tuple containing three list. Each element of those list correspond to a given track and represent : the truth particle ID, the track parameters and the truth.
    @param[in] epochs: number of the current epoch 
    @param[in] batch: size of the batch used in the training
    @param[in] validation: Fraction of the batch used in training
    @param[in] writer: tensorboard writer, optional to keep track of the training
    @param[in] trial: optuna trial, used to keep track of the hyperparameter optimisation (only used in the optimisation mode)
    @return: trained model
    """
    loss = 0.0
    nb_part = 0.0
    nb_good_match = 0.0

        # Compute the score for all the training batch in the epoch
    nb_part, nb_good_match, loss = scoringBatch(
        data, batch[:val_batch], duplicateClassifier, Optimiser=opt
    )
    # Write the result in the tensorboard
    if writer != None:
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Eff/train", nb_good_match / nb_part, epoch)
    # The efficiency of the model is only printed if we are not in optimisation mode
    if trial == None:
        print("Loss/train: ", loss, " Eff/train: ", nb_good_match / nb_part)

    # If using validation, compute the efficiency and loss over the training batch
    if val_batch != len(batch):
        nb_part, nb_good_match, nb_best_match, loss = scoringBatch(
            data, batch[val_batch:], duplicateClassifier
        )
        if writer != None:
            writer.add_scalar("Loss/val", loss, epoch)
            writer.add_scalar("Eff/val", nb_good_match / nb_part, epoch)
        if trial != None:
            # If using optuna, compute the efficiency and loss over the training batch, the trial will be pruned if the efficiency is too low
            trial.report(nb_good_match / nb_part, epoch)
            #if trial.should_prune():
            #    raise optuna.TrialPruned()
        else:
            # The efficiency of the model is only printed if we are not in optimisation mode
            print("Loss/val: ", loss, " Eff/val: ", nb_good_match / nb_part)

    writer.close()
    return nb_good_match / nb_part

def train(
    duplicateClassifier: DuplicateClassifier,
    input: tuple[torch.tensor, torch.tensor, torch.tensor],
    opt: torch.optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 512,
    validation: float = 0.3,
    output: str = "duplicateClassifier",
) -> DuplicateClassifier:
    """Train the model over a number of epochs, it is then written in a file .pt and .onnx files"""
    """
    @param[in] duplicateClassifier: model to be trained.
    @param[in] data: tuple containing three list. Each element of those list correspond to a given seed and represent : the truth particle ID, the seed parameters and the truth.
    @param[in] epochs: number of epoch the model will be trained for.
    @param[in] batch_size: size of the batch used in the training
    @param[in] validation: Fraction of the batch used in training
    @return: trained model
    """
    # Training mode
    duplicateClassifier = duplicateClassifier.to(device)

    # Prepare tensorboard for the training plot
    # use 'tensorboard --logdir=runs' to access the plot afterward
    writer = SummaryWriter("training_ambi_solver")
    # Split the data in batch
    batch = batchSplit(input, batch_size)
    val_batch = int(len(batch) * (1 - validation))
    # Loop over the number of epochs to train the model
    for epoch in range(epochs):
        print("Epoch: ", epoch, " / ", args.epochs)
        score = train_epoch(
            duplicateClassifier,
            opt,
            input,
            epoch=epoch,
            batch=batch,
            val_batch=val_batch,
            writer=writer,
        )
        # Every 10 epoch write the network
        if epoch % 10 == 0:
            torch.save(duplicateClassifier, output + str(epoch + 1) + ".pt")
    duplicateClassifier.eval()
    torch.save(duplicateClassifier, output+".pt")
    torch.onnx.export(
        duplicateClassifier,
        input[1][0:1],
        output+".onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
    )

def test_model(duplicateClassifier: DuplicateClassifier, input: tuple[ np.ndarray, np.ndarray]):
    """Test the model on the test dataset, the efficiency is computed and printed"""
    """
    @param[in] duplicateClassifier: model to be tested.
    @param[in] data: tuple containing three list. Each element of those list correspond to a given track and represent : the truth particle ID, the track parameters and the truth.
    """

    output_predict = []

    x_test = torch.tensor(input[1], dtype=torch.float32)
    x_test = x_test.to(device)
    y_test = input[2]
    output_predict = duplicateClassifier(x_test)

    good_mask = y_test == 0
    duplicate_mask = y_test > 0

    # For the first 100 particles print the ID, score and truth
    for sample_test, sample_predict, sample_true in zip(
        input[0][0:100], output_predict[0:100], y_test[0:100]
    ):
        print(sample_test.item(), sample_predict.item(), sample_true.item())

    boundaries = torch.where(input[0][:-1] != input[0][1:])[0] + 1
    boundaries = boundaries.numpy()

    nb_part = 0
    nb_good_match = 0

    pred = output_predict.cpu().detach().numpy()
    g_mask = good_mask.cpu().detach().numpy()
    d_mask = duplicate_mask.cpu().detach().numpy()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        nb_part += 1
        group_preds = pred[start:end]
        score_good =  group_preds[g_mask[start:end]]
        score_duplicate = group_preds[d_mask[start:end]]
        max_duplicate = score_duplicate.max() if score_duplicate.size > 0 else -1.0
        if score_good > max_duplicate :
            nb_good_match += 1
    print("nb particles: ", nb_part)
    print("nb good match: ", nb_good_match)
    print("nb best match: ", nb_best_match)
    print("Efficiency: ", 100 * nb_good_match / nb_part, " %")

def result_optuna(study):
    """Print the result of the optuna hyperparameter tuning and save the plots"""
    """
    @param[in] study: optuna study containing the result of the hyperparameter tuning
    """

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
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
    """Main function to train the model and optimise the Ranking Based Ambiguity Solver"""

    print("Running on ",device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train", help="Input files for the training", default="odd_output/event0000000[0-7][0-9]-tracks_ckf.csv")
    parser.add_argument("--input_test", help="Input files for the test", default="odd_output/event0000000[8-9][0-9]-tracks_ckf.csv")
    parser.add_argument("--output", help="Output file for the trained model", default="duplicateClassifier")
    parser.add_argument("--epochs", help="Number of epochs", default=100, type=int)
    parser.add_argument("--batch", help="Size of the batch (in truth particles)", default=512, type=int)
    parser.add_argument("--validation", help="Fraction of events udes in validation", default=0.3, type=float)
    parser.add_argument("--optimise", help="Turn on optuna hyperparameter tuning", action='store_true', default=False)
    parser.add_argument("--cpu", help="Force the trainning to be run on CPU", action='store_true', default=False)
    args = parser.parse_args()

    if(args.cpu):
        device = torch.device("cpu")
    # Read the input ttbar event files used as the training input
    CKF_files = sorted(glob.glob(args.input_train))
    data = readDataSet(CKF_files)

    # Prepare the data
    x_train, y_train = prepareTrainingData(data)
    avg_mean = [x / events for x in avg_mean]
    avg_sdv = [x / events for x in avg_sdv]

    id = data.index.str.replace("|", "").astype(int) 

    index_train_t = torch.tensor(id, dtype=torch.int64, requires_grad=False)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)

    input = index_train_t, x_train_t, y_train_t

    if args.optimise:
        # If in optimise mode, try to import optuna
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
            batch_size = args.batch
            validation = args.validation
            epoch_opt = 10

            # Create our model
            input_dim = np.shape(x_train)[1]
            layer_1 = trial.suggest_int("layer_1", 1, 100)
            layer_2 = trial.suggest_int("layer_2", 1, 100)
            layer_3 = trial.suggest_int("layer_3", 1, 100)
            layers_dim = [layer_1, layer_2, layer_3]
            margin = trial.suggest_float("margin", 0.0, 0.4)
            learningRate = trial.suggest_float("learningRate", 0.00001, 1, log=True)

            duplicateClassifier = nn.Sequential(
                Normalise(avg_mean, avg_sdv), DuplicateClassifier(input_dim, layers_dim, margin)
            )
            duplicateClassifier = duplicateClassifier.to(device)

            # Create a writer for the tensorboard base on the trial number
            writer = SummaryWriter("training_ambi_solver/" + str(trial.number))
            opt = torch.optim.Adam(duplicateClassifier.parameters(), lr = learningRate)

            # Split the data in batch
            batch = batchSplit(input, batch_size)
            val_batch = int(len(batch) * (1 - validation))
            # Loop over the number of epochs
            for epoch in range(epoch_opt):
                score = train_epoch(
                    duplicateClassifier, opt, input, epoch = epoch, batch = batch, val_batch = val_batch, writer=writer, trial=trial
                )
            del duplicateClassifier
            return score

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=args.epochs, reduction_factor=3
            ),
        )

        study.optimize(objective, n_trials=200)

        result_optuna(study)

        # Save the best model
        best_trial = study.best_trial
        best_params = best_trial.params
        best_model = nn.Sequential(
            Normalise(avg_mean, avg_sdv),
            DuplicateClassifier(
                np.shape(x_train)[1],
                [best_params["layer_1"], best_params["layer_2"], best_params["layer_3"]],
                best_params["margin"],
            ),
        )
        opt_best = torch.optim.Adam(best_model.parameters(), lr = best_params["learningRate"])
        train(best_model, input, opt_best, epochs=args.epochs, batch_size=args.batch, validation=args.validation, output=args.output)
        # ttbar events for the test, here we assume 1000 events are availables
        CKF_files_test = sorted(glob.glob(args.input_test))
        test = readDataSet(CKF_files_test)

        # Prepare the data
        x_test, y_test = prepareTrainingData(test)

        id = test.index.str.replace("|", "").astype(int) 

        index_test_t = torch.tensor(id, dtype=torch.int64, requires_grad=False)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, requires_grad=False)
        input_test = index_test_t, x_test_t, y_test_t
        test_model(best_model, input_test)

    else:
        # Create our model
        input_dim = np.shape(x_train)[1]
        layers_dim = [15, 20, 15]
        margin = 0.001
        learning_rate = 0.001

        duplicateClassifier = nn.Sequential(
            Normalise(avg_mean, avg_sdv), DuplicateClassifier(input_dim, layers_dim, margin)
        )
        opt = torch.optim.Adam(duplicateClassifier.parameters(), lr = learning_rate)

        # Train the model
        train(duplicateClassifier, input, opt, epochs=args.epochs, batch_size=args.batch, validation=args.validation, output=args.output)

        # ==================================================================
        # ttbar events for the test, here we assume 1000 events are availables
        CKF_files_test = sorted(glob.glob(args.input_test))
        test = readDataSet(CKF_files_test)

        # Prepare the data
        x_test, y_test = prepareTrainingData(test)

        id = test.index.str.replace("|", "").astype(int) 

        index_test_t = torch.tensor(id, dtype=torch.int64, requires_grad=False)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, requires_grad=False)
        input_test = index_test_t, x_test_t, y_test_t
        test_model(duplicateClassifier, input_test)

