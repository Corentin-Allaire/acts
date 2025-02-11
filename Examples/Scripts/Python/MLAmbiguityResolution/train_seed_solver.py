import glob

import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from seed_solver_network import (
    prepareDataSet,
    DuplicateClassifier,
    Normalise,
)

import argparse
import gc

avg_mean = [0] * 14
avg_sdv = [0] * 14
events = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def readDataSet(Seed_files) -> pd.DataFrame:
    """Read the dataset from the different files, remove the particle with only fakes and combine the datasets"""
    """
    @param[in] Seed_files: DataFrame contain the data from each seed files (1 file per events usually)
    @return: combined DataFrame containing all the seed, ordered by events and then by truth particle ID in each events 
    """
    data = pd.DataFrame()
    for f in Seed_files:
        datafile = pd.read_csv(f)
        datafile = prepareDataSet(datafile)
        data = pd.concat([data, datafile])
        print("prepared file:", f)
    return data


def prepareTrainingData(data: pd.DataFrame):
    """Prepare the data"""
    """
    @param[in] data: input DataFrame to be prepared
    @return: array of the network input and the corresponding truth  
    """
    target_column = "rank"
    # Separate the truth from the input variables
    y = data[target_column]
    # Remove truth and useless variable
    input = data.drop(
        columns=[
            target_column,
            "seed_id",
            "Hits_ID",
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
    return batch

def scoringBatch(full_data, batch, duplicateClassifier, Optimiser=0):
    """Run the MLP on a batch and compute the corresponding efficiency and loss. If an optimiser is specify it also trains the MLP."""
    """
    @param[in] batch:  list of DataFrame, each element correspond to a batch 
    @param[in] Optimiser: Optimiser for the MLP, if one is specify the network will be train on batch. 
    @return: array containing the number of particles, the number of particle where the good seed was found and the loss
    """
    # number of particles
    nb_part = 0
    # number of particles associated with a good seed
    nb_good_match = 0
    # number of particles associated with a best seed
    nb_best_match = 0
    # loss for the batch
    loss = torch.tensor(0.0, device=device, requires_grad=False)

    # Extract the trainning part of the data (by oposition to the validation one)
    data = full_data[0][batch[0][0]:batch[-1][-1]], full_data[1][batch[0][0]:batch[-1][-1]], full_data[2][batch[0][0]:batch[-1][-1]]
    # Compute mask to select good, duplicate or fake
    good_mask = data[2] == 0
    duplicate_mask = data[2] > 0
    fake_mask = data[2] < 0
    # Compute tensors that will be used to compute the loss as a single matrix multiplication
    rank = torch.clone(data[2])
    good = torch.zeros_like(data[2])
    rank[fake_mask] = 5
    good[fake_mask] = 0
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
        truth_good = good[id_cut:id_next]

        good_score = predictions[good_mask[id_cut:id_next]]
        nb_part += len(good_score)
        nb_seed = torch.tensor([end-start for start, end in zip(boundaries[:-1], boundaries[1:])], device=device, requires_grad=False)
        good_score = good_score.repeat_interleave(nb_seed)
        # Compute a loss with 2 terms, a margin ranking loss to separate the good from duplicate 
        # and binary cross entropy to separate the fake from the rest
        batch_loss = torch.tensor(0.0, requires_grad=False, device=device)
        batch_loss += F.relu(predictions - good_score + truths_rank * duplicateClassifier[1].marginDuplicate).sum()
        batch_loss += torch.nn.functional.binary_cross_entropy(predictions, truth_good)

        # Normalize loss and accumulate
        batch_loss = batch_loss / len(predictions)
        loss += batch_loss
        # Perform gradient update if Optimiser is specified
        if Optimiser:
            batch_loss.backward()
            Optimiser.step()
            # used to ignore the efficiency computation in the training loop 
            nb_good_match = nb_part
            nb_best_match = nb_part
        else:
            # Compute the efficiency in the validation pass
            # This is perform on CPU with numpy array as it is much faster
            pred = predictions.cpu().detach().numpy()
            g_mask = good_mask.cpu().detach().numpy()
            d_mask = duplicate_mask.cpu().detach().numpy()
            f_mask = fake_mask.cpu().detach().numpy()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                group_preds = pred[start:end]
                score_good =  group_preds[g_mask[id_cut+start:id_cut+end]]
                score_duplicate = group_preds[d_mask[id_cut+start:id_cut+end]]
                score_fake = group_preds[f_mask[id_cut+start:id_cut+end]]
                max_duplicate = score_duplicate.max() if score_duplicate.size > 0 else -1.0
                max_fake = score_fake.max() if score_fake.size > 0 else -1.0
                if score_good > max_fake or max_duplicate > max_fake:
                    nb_good_match += 1
                if score_good > max_duplicate and score_good > max_fake:
                    nb_best_match += 1

    loss = loss.item() / len(batch)
    return nb_part, nb_good_match, nb_best_match, loss

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
    @param[in] epoch: number of the current epoch 
    @param[in] batch: size of the batch used in the training
    @param[in] validation: Fraction of the batch used in training
    @param[in] writer: tensorboard writer, optional to keep track of the training
    @param[in] trial: optuna trial, used to keep track of the hyperparameter optimisation (only used in the optimisation mode)
    @return: Efficiency over the training batch, used as score for the hyperparameter optimisation
    """
    
    # Loop over all the epoch
    loss = 0.0
    nb_part = 0.0
    nb_good_match = 0.0
    nb_best_match = 0.0
    # Compute the score for all the training batch in the epoch
    nb_part, nb_good_match, nb_best_match, loss = scoringBatch(
        data, batch[:val_batch], duplicateClassifier, Optimiser=opt
    )

    if trial == None:
        print(
            "Loss/train: ",
            loss,
        )
    if writer != None:
        writer.add_scalar("Loss/train", loss, epoch)

    # If using validation, compute the efficiency and loss over the training batch
    if val_batch != len(batch):
        nb_part, nb_good_match, nb_best_match, loss = scoringBatch(
            data, batch[val_batch:], duplicateClassifier
        )
        if writer != None:
            writer.add_scalar("Loss/val", loss, epoch)
            writer.add_scalar("Eff/val", nb_good_match / nb_part, epoch)
            writer.add_scalar("Eff_best/train", nb_best_match / nb_part, epoch)
        if trial != None:
            trial.report( (nb_good_match / nb_part)*10 + (nb_best_match / nb_part) , epoch)
            # Trial prunning is currently disable as it removed too many (good) trials
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
        else:
            print(
                "Loss/val: ",
                loss,
                " Eff/val: ",
                nb_good_match / nb_part,
                " Eff_best/val: ",
                nb_best_match / nb_part,
            )
    # An efficiency based score is computed for each epoch.
    # It correspond to 10*good_match_eff + best_match_eff (between 0 and 1.1)
    return ((nb_good_match / nb_part)*10 + (nb_best_match / nb_part))

def train(
    duplicateClassifier: DuplicateClassifier,
    input: tuple[torch.tensor, torch.tensor, torch.tensor],
    opt: torch.optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 512,
    validation: float = 0.3,
    output: str = "seedFilter",
) -> DuplicateClassifier:
    """Training of the MLP"""
    """
    @param[in] duplicateClassifier: model to be trained.
    @param[in] data: tuple containing three list. Each element of those list correspond to a given seed and represent : the truth particle ID, the seed parameters and the truth.
    @param[in] epochs: number of epoch the model will be trained for.
    @param[in] batch_size: size of the batch used in the training
    @param[in] validation: Fraction of the batch used in training
    @return: trained model
    """
    # Transfer the network to the device
    duplicateClassifier = duplicateClassifier.to(device)

    # Prepare tensorboard to get training plots
    # use 'tensorboard --logdir=runs' to access the plot afterward
    writer = SummaryWriter("training_seed_filter")

    # Create a learning rate scheduler that will pregressively decrease its value down to 0.05 the original one 
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.05, total_iters=max(300, 0.75*epochs))
    
    # Split the data in batch
    batch = batchSplit(input, batch_size)
    val_batch = int(len(batch) * (1 - validation))
    best_score = 0
    # Loop over all the epochs
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
        # Update the learning rate
        scheduler.step()
        # Every 10 epoch write the network
        if epoch % 10 == 0:
            torch.save(duplicateClassifier, output + str(epoch + 1) + ".pt")
        # Also write the best network so far (in term of efficiency)
        if best_score < score:
            best_score = score
            torch.save(duplicateClassifier, output + "_best.pt")
            torch.onnx.export(
                duplicateClassifier,
                input[1][0:1],
                output+"_best.onnx",
                input_names=["x"],
                output_names=["y"],
                dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
            )
    # Save the final network and export it to onnx
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
    del batch

def test_model(
    duplicateClassifier: DuplicateClassifier, input: tuple[torch.tensor, torch.tensor, torch.tensor]
):

    x_test = torch.tensor(input[1], dtype=torch.float32)
    x_test = x_test.to(device)
    y_test = input[2]
    output_predict = duplicateClassifier(x_test)

    good_mask = y_test == 0
    duplicate_mask = y_test > 0
    fake_mask = y_test < 0

    # For the first 100 particles print the ID, score and truth
    for sample_test, sample_predict, sample_true in zip(
        input[0][0:100], output_predict[0:100], y_test[0:100]
    ):
        print(sample_test.item(), sample_predict.item(), sample_true.item())

    boundaries = torch.where(input[0][:-1] != input[0][1:])[0] + 1
    boundaries = boundaries.numpy()

    nb_part = 0
    nb_good_match = 0
    nb_best_match = 0

    pred = output_predict.cpu().detach().numpy()
    g_mask = good_mask.cpu().detach().numpy()
    d_mask = duplicate_mask.cpu().detach().numpy()
    f_mask = fake_mask.cpu().detach().numpy()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        nb_part += 1
        group_preds = pred[start:end]
        score_good =  group_preds[g_mask[start:end]]
        score_duplicate = group_preds[d_mask[start:end]]
        score_fake = group_preds[f_mask[start:end]]
        max_duplicate = score_duplicate.max() if score_duplicate.size > 0 else -1.0
        max_fake = score_fake.max() if score_fake.size > 0 else -1.0
        if score_good > max_fake or max_duplicate > max_fake:
            nb_good_match += 1
        if score_good > max_duplicate and score_good > max_fake:
            nb_best_match += 1

    print("nb particles: ", nb_part)
    print("nb good match: ", nb_good_match)
    print("nb best match: ", nb_best_match)
    print("Efficiency: ", 100 * nb_good_match / nb_part, " %")
    print("Efficiency_best: ", 100 * nb_best_match / nb_part, " %")

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
    """Main function to train the model and optimise the Ranking Based Ambiguity Solver"""

    print("Running on ",device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_train",
        help="Input files for the training",
        default="odd_output/event0000000[0-7][0-9]-seed_cleaned.csv",
    )
    parser.add_argument("--input_test",help="Input files for the test",default="odd_output/event0000000[8-9][0-9]-seed_cleaned.csv")
    parser.add_argument("--output", help="Output file for the trained model", default="duplicateClassifier")
    parser.add_argument("--epochs", help="Number of epochs", default=100, type=int)
    parser.add_argument("--batch", help="Size of the batch (in truth particles)", default=512, type=int)
    parser.add_argument("--validation", help="Fraction of events udes in validation", default=0.3, type=float)
    parser.add_argument("--optimise", help="Turn on optuna hyperparameter tuning", action='store_true', default=False)
    parser.add_argument("--optimise_layer", help="number of layer in optimisation", default=5, type=int)    
    parser.add_argument("--load_tensor", help="directly load the torch tensor from files ignoring pre-processing", action='store_true', default=False)
    parser.add_argument("--input_tensor", help="path to the pre-processed tensor (for writing or reading depemding on load_tensor)", default="odd_output/")
    parser.add_argument("--cpu", help="Force the trainning to be run on CPU", action='store_true', default=False)

    args = parser.parse_args()

    if(args.cpu):
        device = torch.device("cpu")

    if(args.load_tensor):
        # Input data is store as tensor after preprocessing to speed up the input access
        print("Loading training tensor from files")
        index_train_t = torch.load(args.input_tensor + "seed_index_t.pt", weights_only=True)
        x_train_t = torch.load(args.input_tensor + "seed_x_train_t.pt", weights_only=True)
        y_train_t = torch.load(args.input_tensor + "seed_y_train_t.pt", weights_only=True)

        index_test_t = torch.load(args.input_tensor + "seed_index_test_t.pt", weights_only=True)
        x_test_t = torch.load(args.input_tensor + "seed_x_test_t.pt", weights_only=True)
        y_test_t = torch.load(args.input_tensor + "seed_y_test_t.pt", weights_only=True)

        avg_mean_t = torch.load(args.input_tensor + "seed_avg_mean_t.pt", weights_only=True)
        avg_sdv_t = torch.load(args.input_tensor + "seed_avg_sdv_t.pt", weights_only=True)
    else:
        # Read the input ttbar event files used as the training input
        seed_input = sorted(glob.glob(args.input_train))
        print(seed_input)
        data = readDataSet(seed_input)
        # Prepare the data
        x_train, y_train = prepareTrainingData(data)

        avg_mean = [x / events for x in avg_mean]
        avg_sdv = [x / events for x in avg_sdv]

        # Transfor the particle ID to a single number
        id = data.index.str.replace("|", "").astype(int) 

        index_train_t = torch.tensor(id, dtype=torch.int64, requires_grad=False)
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)

        avg_mean_t = torch.tensor(avg_mean, dtype=torch.float32) 
        avg_sdv_t = torch.tensor(avg_sdv, dtype=torch.float32)

        torch.save(index_train_t, args.input_tensor + "seed_index_t.pt")
        torch.save(x_train_t, args.input_tensor + "seed_x_train_t.pt")
        torch.save(y_train_t, args.input_tensor + "seed_y_train_t.pt")

        torch.save(avg_mean_t, args.input_tensor + "seed_avg_mean_t.pt")
        torch.save(avg_sdv_t, args.input_tensor + "seed_avg_sdv_t.pt")

        print("Training tensor have been written at ", args.input_tensor)

        del seed_input
        del data
        del x_train
        del y_train
        gc.collect()

        # Read the input ttbar event files used as the test input
        seed_input = sorted(glob.glob(args.input_test))
        data = readDataSet(seed_input)
        # Prepare the data
        x_test, y_test = prepareTrainingData(data)

        # Transfor the particle ID to a single number
        id = data.index.str.replace("|", "").astype(int)

        index_test_t = torch.tensor(id, dtype=torch.int64, requires_grad=False)
        x_test_t = torch.tensor(x_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, requires_grad=False)

        torch.save(index_test_t, args.input_tensor + "seed_index_test_t.pt")
        torch.save(x_test_t, args.input_tensor + "seed_x_test_t.pt")
        torch.save(y_test_t, args.input_tensor + "seed_y_test_t.pt")

        print("Testing tensor have been written ", args.input_tensor)

        del seed_input
        del data
        del x_test
        del y_test
        gc.collect()
    
    # Move training input and truth to device
    x_train_t = x_train_t.to(device)
    y_train_t = y_train_t.to(device)
    input = index_train_t, x_train_t, y_train_t

    # Move testing input and truth to device
    x_test_t = x_test_t.to(device)
    y_test_t = y_test_t.to(device)
    input_test = index_test_t, x_test_t, y_test_t

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
            batch_size = args.batch
            validation = args.validation
            epoch_opt = 10

            # Create our model
            input_dim = np.shape(x_train_t)[1]
            nb_layer = args.optimise_layer
            layer = []
            for i_layer in range(nb_layer):
                layer.append(trial.suggest_int(("layer_" + str(i_layer)), 10, 100))
            marginDuplicate = trial.suggest_float("marginDuplicate", 0.0, 1)
            learningRate = trial.suggest_float("learningRate", 0.00001, 1, log=True)
            # learningRate = 0.002
            # marginDuplicate = 0.1
            duplicateClassifier = nn.Sequential(
                Normalise(avg_mean_t, avg_sdv_t), DuplicateClassifier(input_dim, layer, marginDuplicate)
            )
            duplicateClassifier = duplicateClassifier.to(device)
            opt = torch.optim.Adam(duplicateClassifier.parameters(), lr = learningRate)
            
            # Create a writer for the tensorboard base on the trial number
            writer = SummaryWriter("training_seed_filter/" + str(trial.number))

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
               min_resource=1, max_resource=10, reduction_factor=3
            ),
        )

        study.optimize(objective, n_trials=5)

        result_optuna(study)

        # Run the full trinning with the best possible model
        best_trial = study.best_trial
        best_params = best_trial.params
        best_layer = []
        for i_layer in range(args.optimise_layer):
            best_layer.append(best_params["layer_" + str(i_layer)])
        best_model = nn.Sequential(
            Normalise(avg_mean, avg_sdv),
            DuplicateClassifier(
                np.shape(x_train_t)[1],
                best_layer,
                best_params["marginDuplicate"],
            )
        )
        opt_best = torch.optim.Adam(best_model.parameters(), lr = best_params["learningRate"])
        train(best_model, input, opt_best, epochs=args.epochs, batch_size=args.batch, validation=args.validation, output=args.output)

        del x_train_t
        del y_train_t
        del index_train_t
        del input
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test the modele on the test dataset
        test_model(best_model, input_test) 

    else:
        # Define the parameters of the modele
        input_dim = x_train_t.shape[1]
        # layers_dim = [200, 150, 100, 80, 40]        
        layers_dim = [80, 50, 80]
        margin = 0.017
        learning_rate = 0.017
        # Create the modele
        duplicateClassifier = nn.Sequential(
            Normalise(avg_mean_t, avg_sdv_t), DuplicateClassifier(input_dim, layers_dim, margin)
        )
        opt = torch.optim.Adam(duplicateClassifier.parameters(), lr = learning_rate)

        # Train the modele
        train(duplicateClassifier, input, opt, epochs=args.epochs, batch_size=args.batch, validation=args.validation, output=args.output)

        del x_train_t
        del y_train_t
        del index_train_t
        del input
        gc.collect()
        if torch.cuda.is_available():
            print("empty cuda cache")
            torch.cuda.empty_cache()

        # Test the modele on the test dataset
        test_model(duplicateClassifier, input_test) 
