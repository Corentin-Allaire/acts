import pandas as pd

import torch.nn.functional as F
import torch.utils
from torch import Tensor

import matplotlib.pyplot as plt

from transformer_network import SeedTransformer
from mask_network import build_look_ahead_mask
from embedding_network import (
    EmbeddingGeoID,
    EmbeddingHitPosition,
    EmbeddingHitIDPosition,
)

import sys
from typing import Tuple
import argparse
import os
import math


def init_seed(size: int, device_acc: str) -> Tensor:
    """
    Initialise the seed input for the decoder, the seed is a tensor of size (batch_size, 1, 5)
    The 7 values are the vertex z0, the momentum phi, eta, pT and the iter value, they are all set to 0
    except for the iter value which is set to 1 and the pT with is set to -1
    Args:
        - size: the batch size for the decoder
        - device_acc: the device to use (cpu/gpu)
    Returns:
        - Tensor: The initial seed tensor
    """
    initial_seed = torch.zeros(size, 1, 5)
    initial_seed[:, 0, 3] = -1
    initial_seed[:, 0, 4] = 1
    initial_seed.to(device_acc)
    return initial_seed


def plot_loss(
    metrics_train: list[float],
    metrics_val: list[float],
    loss_name: str,
    interactive: bool = True,
):
    """
    Plot the loss of the training and validation in different color of the epoch using matplotlib
    Args:
        - metrics_train: the list of the loss for the training
        - metrics_val: the list of the loss for the validation
        - loss_name: the name of the loss
        - interactive: if set to True the plot will be displayed if not it will only be saved as a png file
    """
    plt.plot(metrics_train, label="Training")
    plt.plot(metrics_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)
    plt.legend()
    plt.savefig(loss_name + ".png")
    if interactive:
        plt.show()
    pd.DataFrame(metrics_train).to_csv(loss_name + "_train.csv")
    pd.DataFrame(metrics_val).to_csv(loss_name + "_val.csv")


class config:
    """
    Class to store the configuration variable of the training
    """

    def __init__(self):
        """
        Initialise the configuration
        Members:
            - embedding: str: The type of embedding to use
            - epoch_nb: int: The number of epoch
            - batch_size: int: The batch size
            - max_hit_input: int: The maximum number of hit in the encoder input
            - max_particle_input: int: The maximum number of particle in the decoder input
            - vertex_cuts: list[int]: The cuts to apply on the vertex position to only keep the primary vertex
            - event_test: int: Number of event in the test will be run on
            - interactive: bool: If set to True the plot will be displayed if not it will only be saved as a png file
            - encoder_only: bool: If set to True only the encoder will be run (can be use to pretrain the network)
            - input_type: str: The type of input to use (csv or tensor)
            - device_acc: str: The device to use (cpu/gpu)
        """
        self.embedding = "ID+Position"
        self.epoch_nb = 100
        self.batch_size = 10
        self.max_hit_input = 2048
        self.max_particle_input = 200
        self.vertex_cuts = [10, 10, 200]
        self.event_test = 1
        self.interactive = True
        self.encoder_only = False
        self.input_type = "csv"
        self.device_acc = torch.device("cpu")

    def parse_args(self):
        """
        Parse the command line argument to fill the configuration
        """
        parser = argparse.ArgumentParser(
            description="Fill the config from the command line"
        )
        parser.add_argument(
            "--vertex_cuts", type=list, default=[10, 10, 200], help="Vertex cuts"
        )
        parser.add_argument("--epoch_nb", type=int, default=10, help="Number of epoch")
        parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
        parser.add_argument(
            "--max_hit_input",
            type=int,
            default=2048,
            help="Maximum number of hit input",
        )
        parser.add_argument(
            "--max_particle_input",
            type=int,
            default=100,
            help="Maximum number of particle input",
        )
        parser.add_argument("--event_test", type=int, default=1, help="Event to test")
        parser.add_argument(
            "--encoder_only", type=bool, default=False, help="Encoder only"
        )
        parser.add_argument(
            "--interactive", type=bool, default=True, help="Interactive plot"
        )
        parser.add_argument(
            "--embedding", type=str, default="ID", help="Type of embedding"
        )
        parser.add_argument(
            "--input_type", type=str, default="csv", help="Type of input"
        )
        args = parser.parse_args()
        self.vertex_cuts = args.vertex_cuts
        self.epoch_nb = args.epoch_nb
        self.batch_size = args.batch_size
        self.max_hit_input = args.max_hit_input
        self.max_particle_input = args.max_particle_input
        self.event_test = args.event_test
        self.interactive = args.interactive
        self.embedding = args.embedding
        self.encoder_only = args.encoder_only
        self.input_type = args.input_type

    def print_config(self):
        """
        Print the configuration
        """
        print("The embedding used is", self.embedding)
        print("The number of epoch is", self.epoch_nb)
        print("The batch size is", self.batch_size)
        print("The max number of hit input is", self.max_hit_input)
        print("The max number of particle input is", self.max_particle_input)


class metrics:
    """
    Class to store the loss (and other performance metrics) of the training and validation
    """

    def __init__(self, epoch_nb: int = 1):
        """
        Initialise the metrics
        Args:
            - epoch_nb: the number of epoch
        Members:
            - loss: list[float]: the loss for each epoch
            - loss_nb: list[float]: the loss for the number of seed computed by the decoder for each epoch
            - loss_momentum: list[float]: the loss for the momentum computed by the decoder for each epoch
            - loss_iter: list[float]: the loss for the iter variable computed by the decoder for each epoch
        """
        self.loss = [0] * epoch_nb
        self.loss_class = [0] * epoch_nb
        self.loss_momentum = [0] * epoch_nb
        self.loss_iter = [0] * epoch_nb

    def add_loss(
        self,
        epoch: int,
        loss: float,
        loss_class: float,
        loss_momentum: float,
        loss_iter: float,
    ):
        """
        Add the loss for a given epoch
        Args:
            - epoch: the epoch number
            - loss: the loss value
            - loss_nb: the loss for the number of seed computed by the decoder
            - loss_momentum: the loss for the momentum computed by the decoder
            - loss_iter: the loss for the iter variable computed by the decoder
        """
        self.loss[epoch] += loss
        self.loss_class[epoch] += loss_class
        self.loss_momentum[epoch] += loss_momentum
        self.loss_iter[epoch] += loss_iter

    def print_loss(self, epoch: int):
        """
        Print the loss for a given epoch
        Args:
            - epoch: the epoch number
        """
        print("Epoch", epoch, "loss is", self.loss[epoch])
        print("Epoch", epoch, "loss_class is", self.loss_class[epoch])
        print("Epoch", epoch, "loss_momentum is", self.loss_momentum[epoch])
        print("Epoch", epoch, "loss_iter is", self.loss_iter[epoch])


def prepare_input_tensor(
    hits: pd.DataFrame,
    particles: pd.DataFrame,
    nb_events: int,
    cfg: config,
    embedding,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Prepare the input tensor and padding mask for a given event.

    Args:
        - hits: the input hits DataFrame
        - particles: the input particles DataFrame
        - nb_events: the number of events to be processed
        - cfg: the configuration of the training
    Returns:
        - A tuple containing the hits tensor, the particles tensor, the number of particles tensor, the padding mask for the hits and the padding mask for the particles
    """
    # Ininitalise the padding masks
    padding_mask_hit = torch.zeros(nb_events, cfg.max_hit_input)
    padding_mask_particle = torch.zeros(nb_events, cfg.max_particle_input)
    particle_class = torch.zeros(nb_events, cfg.max_hit_input, dtype=torch.long)
    for i in range(nb_events):
        # Select the hits and particles for the event i
        hits_event = hits[hits["event_id"] == i]
        particles_event = particles[particles["event_id"] == i]

        # particle_class is a tensor of size (nb_events, nb_hits) for each hit it associate a particle id between 0 and max_particle_input

        # Map the particle id to a class between 0 and max_particle_input
        map_id = {}
        for j, particle_id in enumerate(particles_event["particle_id"].unique()):
            if j < cfg.max_particle_input:
                map_id[particle_id] = j + 1
            else:
                map_id[particle_id] = 0

        particle_class[i, : len(hits_event)] = torch.tensor(
            hits_event["particle_id"].map(lambda x: map_id.get(x, 0)).values
        )

        particles_event = particles_event[["vz", "eta", "phi", "pT"]]
        # Add one column to the particles DataFrame to indicate if a particle is the last one in the events
        particles_event["iter"] = 1
        particles_event.iloc[-1, -1] = 0
        # Depending on the embedding used, select the right columns from the hits
        if type(embedding) == EmbeddingGeoID:
            tensor_hit = torch.tensor(
                hits_event[["volume", "layer", "sensitive", "extra"]].values,
                dtype=torch.float32,
            )
        elif type(embedding) == EmbeddingHitPosition:
            tensor_hit = torch.tensor(
                hits_event[["tx", "ty", "tz"]].values, dtype=torch.float32
            )
        elif type(embedding) == EmbeddingHitIDPosition:
            tensor_hit = torch.tensor(
                hits_event[["volume", "layer", "tx", "ty", "tz"]].values,
                dtype=torch.float32,
            )
        tensor_particle = torch.tensor(particles_event.values, dtype=torch.float32)

        # Pad the tensors to match the input size
        input_size_hits = tensor_hit.size(0)
        padding_mask_hit[i, input_size_hits:] = 1
        if input_size_hits > cfg.max_hit_input:
            tensor_hit = tensor_hit[: cfg.max_hit_input, :]
            print(
                "The number of hits is greater than",
                cfg.max_hit_input,
                ", this hits further away from he center have been removed but this might cause issues !!!",
            )
        else:
            # ID embedding : pad with zeros
            if type(embedding) == EmbeddingGeoID:
                tensor_hit = torch.cat(
                    (tensor_hit, torch.zeros(cfg.max_hit_input - input_size_hits, 4))
                )
            # hit embedding : pad with the minimal value of the corresponding range : -3000 (bin 0)
            elif type(embedding) == EmbeddingHitPosition:
                padding = torch.cat(
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_x[0]
                    ),
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_y[0]
                    ),
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_z[0]
                    ),
                    dim=1,
                )
                tensor_hit = torch.cat((tensor_hit, padding), dim=0)
            # hit+ID embedding : pad with the minimal value of the corresponding range : -3000 (bin 0) for the position and 0 for the ID
            elif type(embedding) == EmbeddingHitIDPosition:
                padding = torch.cat(
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_x[0]
                    ),
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_y[0]
                    ),
                    torch.full(
                        (cfg.max_hit_input - input_size_hits, 1), embedding.range_z[0]
                    ),
                    torch.zeros(cfg.max_hit_input - input_size_hits, 2),
                    dim=1,
                )
                tensor_hit = torch.cat((tensor_hit, padding), dim=0)
        # Pad the input particle tensor to match the input size
        input_size_particle = tensor_particle.size(0)
        padding_mask_particle[i, input_size_particle:] = 1
        if input_size_particle > cfg.max_particle_input:
            tensor_particle = tensor_particle[: cfg.max_particle_input, :]
            print(
                "The number of particles is greater than",
                cfg.max_particle_input,
                ", the particles with the smallest momentum have been removed but this might cause issues !!!",
            )
        else:
            tensor_particle = torch.cat(
                (
                    tensor_particle,
                    torch.zeros(cfg.max_particle_input - input_size_particle, 5),
                )
            )

        if i == 0:
            input_tensor_hits = tensor_hit.unsqueeze(0)
            input_tensor_particles = tensor_particle.unsqueeze(0)

        else:
            input_tensor_hits = torch.cat((input_tensor_hits, tensor_hit.unsqueeze(0)))
            input_tensor_particles = torch.cat(
                (input_tensor_particles, tensor_particle.unsqueeze(0))
            )

    return (
        input_tensor_hits,
        input_tensor_particles,
        particle_class,
        padding_mask_hit,
        padding_mask_particle,
    )


def read_data(
    file_hits: str,
    file_particles: str,
    start_event: int = 0,
    vertex_cuts: list = [10, 10, 200],
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Read the hits and particles csv files and return the DataFrames and the number of events in the dataset dataset after cuts
    Args:
        - file_hits: the file containing the hits
        - file_particles: the file containing the particles
        - vertex_cuts: the cuts to apply on the vertex position to only keep the primary vertex
    Returns:
        - A tuple containing the hits DataFrame, the particles DataFrame and the number of events in the dataset after cuts
    """
    # Open the hits and particles csv files
    hits = pd.read_csv(file_hits)
    particles = pd.read_csv(file_particles)

    # Remove all particle not coming from a primary vertex
    particles = particles[abs(particles["vx"]) < vertex_cuts[0]]
    particles = particles[abs(particles["vy"]) < vertex_cuts[1]]
    particles = particles[abs(particles["vz"]) < vertex_cuts[2]]

    # Remove all events with no particles
    hits = hits[hits["event_id"].isin(particles["event_id"].unique())]
    # Modify the value of events so it is continuous again
    events = particles["event_id"].unique()
    for i in range(len(events)):
        particles["event_id"] = particles["event_id"].replace(
            events[i], start_event + i
        )
        hits["event_id"] = hits["event_id"].replace(events[i], start_event + i)

    nb_events = len(hits["event_id"].unique())
    return hits, particles, nb_events


def decoder_loss(
    momentum: Tensor,
    keep_iterating: Tensor,
    particles: Tensor,
    enrich_iter: int,
) -> Tensor:
    """
    Compute the 2 loss for the decoder: the loss for the momentum and the loss for the iter value
    Args:
        - momentum: the momentum computed by the decoder
        - keep_iterating: the iter value computed by the decoder
        - particles: the particles tensors
        - enrich_iter: value between 0 and 1 to balance the case iter=0 and iter=1
    """
    if enrich_iter < 0 or enrich_iter > 1:
        print("The enrich_iter value should be between 0 and 1")
        exit()

    # Compute the loss by comparing the decoded value to the particles in the event
    loss_momentum = F.mse_loss(momentum, particles[:, :, 0:4])
    loss_iter = F.binary_cross_entropy(keep_iterating, particles[:, :, 4].unsqueeze(-1))

    # Extract the keep_iterating value for the last particle in the event (it should be 0)
    stop_iterating = keep_iterating - (
        keep_iterating * particles[:, :, 4].unsqueeze(-1)
    )
    # Compute the corresponding loss, enrich_iter will be used to balance the case iter=0 and iter=1
    loss_iter = (1 - enrich_iter) * loss_iter + enrich_iter * 100 * F.mse_loss(
        torch.zeros_like(stop_iterating), stop_iterating
    )

    return loss_momentum, loss_iter


def compute_loss(
    hits: Tensor,
    particles: Tensor,
    padding_mask_hits: Tensor,
    padding_mask_particle: Tensor,
    particle_class: Tensor,
    mask_hits: Tensor,
    mask_particle: Tensor,
    initial_seed: Tensor,
    model: SeedTransformer,
    device: str,
    encoder_only: bool = False,
) -> Tensor:
    """
    Run the encoder and decoder on the hits and particles and compute the associated loss for one batch
    Args:
        - hits: the hits tensor
        - particles: the particles tensor
        - padding_mask_hits: the padding mask for the hits (encoder)
        - padding_mask_particle: the padding mask for the particles (decoder)
        - nb_particles: the number of particles in the event
        - mask_hits: the look ahead mask for the encoder
        - mask_particle: the look ahead mask for the decoder (used in the training to avoid the looping part of the inference)
        - initial_seed: the initial seed for the decoder
        - model: the transformer model
    Returns:
        - A tuple containing the loss for the number of seed, the loss for the momentum and the loss for the iter value
    """
    # Run the encoder on the hits
    # Print the device of the encoder input
    print("The device of the encoder input is", device)
    # Add all the input tensor to the device
    hits = hits.to(device)
    padding_mask_hits = padding_mask_hits.to(device)
    padding_mask_particle = padding_mask_particle.to(device)
    particle_class = particle_class.to(device)

    encoded = model.encode(hits, mask_hits, padding_mask_hits)
    # Compute the loss for nb_seed by comparing its value to the number of particles in the events of the batch

    loss_class = 0
    # Using the CosineEmbeddingLoss to compute the loss for the encoding of the hits (encoded)
    # This will be used to ensure that the encoding of hit belonging to the same particle are close to each other
    # and the encoding of hit belonging to different particle are far from each other
    # The target is set to 1 for hit belonging to the same particle and -1 for hit belonging to different particle

    for i in range(particle_class.size(0)):
        # count the number of 1 in padding_mask_hits
        non_pad = torch.sum(1 - padding_mask_hits[i])
        for j in range(non_pad):
            # Créer un masque pour les éléments égaux
            mask = (particle_class[i, j] == particle_class[i, j + 1 :]).float()
            target = mask * 2 - 1  # Convertir True/False en 1/-1

            # Calculer la perte en une seule opération vectorisée
            lo = F.cosine_embedding_loss(
                encoded[i, j].unsqueeze(0).expand_as(encoded[i, j + 1 :]),
                encoded[i, j + 1 :],
                target,
            )
            loss_class += lo

    if encoder_only == True:
        loss_momentum, loss_iter = (
            torch.zeros(1),
            torch.zeros(1),
        )
    else:
        # We will now run the decoder on the particles
        # Create the decoder input by concatenating the initial seed with the particles
        # The last particle in the event is not added since the decoder should not run on it
        input = torch.cat((initial_seed, particles[:, :-1, :]), dim=1)

        momentum, keep_iterating = model.decode(
            input, encoded, mask_particle, padding_mask_particle
        )

        # set to 0 all the element of elements that are not in the nb_particles
        momentum = momentum - momentum * padding_mask_particle.unsqueeze(-1)
        keep_iterating = (
            keep_iterating - keep_iterating * padding_mask_particle.unsqueeze(-1)
        )

        # Compute the decoder loss
        loss_momentum, loss_iter = decoder_loss(
            momentum,
            keep_iterating,
            particles,
            0.5,
        )
    return loss_class, loss_momentum, loss_iter


def run_model(
    epoch: int,
    cfg: config,
    input_tensor_hits: Tensor,
    input_tensor_particles: Tensor,
    particle_class: Tensor,
    padding_mask_hit: Tensor,
    padding_mask_particle: Tensor,
    model: SeedTransformer,
    met: metrics,
    optimiser: torch.optim.Optimizer = None,
) -> SeedTransformer:
    """
    Run the model for training or validation
    Args:
        - hits: The hits DataFrame.
        - particles: The particles DataFrame.
        - nb_events: The number of events in the dataset.
        - model: The transformer model.
        - optimiser: The optimiser to use for training
    Returns:
        - The trained transformer model.
    """

    nb_batches = input_tensor_hits.size(0) // cfg.batch_size
    # Loop over the event batches
    for i in range(nb_batches):
        if i % 100 == 0:
            if optimiser is not None:
                print("Training batch:", i, "/", nb_batches)
            else:
                print("Validation batch:", i, "/", nb_batches)

        # Select the batch of hits and particles
        batch_tensor_hits = input_tensor_hits[
            i * cfg.batch_size : (i + 1) * cfg.batch_size
        ]
        batch_tensor_particles = input_tensor_particles[
            i * cfg.batch_size : (i + 1) * cfg.batch_size
        ]
        batch_padding_hit = padding_mask_hit[
            i * cfg.batch_size : (i + 1) * cfg.batch_size
        ]
        batch_padding_particle = padding_mask_particle[
            i * cfg.batch_size : (i + 1) * cfg.batch_size
        ]
        batch_particle_class = particle_class[
            i * cfg.batch_size : (i + 1) * cfg.batch_size
        ]
        # Create the lookahead mask for the hit (encoder) and particle (decoder)
        mask_hits = build_look_ahead_mask(cfg.max_hit_input, cfg.device_acc)
        mask_particle = build_look_ahead_mask(cfg.max_particle_input, cfg.device_acc)
        # Create an initialisation seed for the transformer
        initial_seed = init_seed(cfg.batch_size, cfg.device_acc)
        # Compute the loos for the batch
        loss_class, loss_momentum, loss_iter = compute_loss(
            batch_tensor_hits,
            batch_tensor_particles,
            batch_padding_hit,
            batch_padding_particle,
            batch_particle_class,
            mask_hits,
            mask_particle,
            initial_seed,
            model,
            cfg.device_acc,
            cfg.encoder_only,
        )
        if i % 100 == 0:
            print("The loss class: ", loss_class.item())
            print("The loss momentum: ", loss_momentum.item())
            print("The loss iter: ", loss_iter.item())

        # Add the loss to t
        loss = 1 * loss_class + 0.1 * loss_momentum + 10 * loss_iter
        met.add_loss(
            epoch,
            loss.item(),
            loss_class.item(),
            loss_momentum.item(),
            loss_iter.item(),
        )
        # If we are training, backpropagate the loss
        if optimiser is not None:
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        # # in case of debugging, only run the first 10 events
        # if i == 3:
        #     break

    return model, met


def test_model(
    cfg: config,
    hits: pd.DataFrame,
    particles: pd.DataFrame,
    nb_events: int,
    model: SeedTransformer,
):
    """
    Test the model on the test dataset
    Args:
        - cfg: the configuration of the training
        - hits: the hits DataFrame
        - particles: the particles DataFrame
        - nb_events: the number of events in the dataset
        - model: the transformer model
    """
    # Prepare the input tensor and padding mask
    (
        input_tensor_hits,
        input_tensor_particles,
        particle_class,
        padding_mask_hit,
        _,
    ) = prepare_input_tensor(hits, particles, nb_events, cfg, model.embedding_encoder)

    # Create the look ahead mask
    mask_hits = build_look_ahead_mask(cfg.max_hit_input, cfg.device_acc)
    mask_particles = build_look_ahead_mask(cfg.max_particle_input, cfg.device_acc)
    initial_seed = init_seed(1, cfg.device_acc)
    # Pad the initial seed up to the maximum number of seed
    initial_seed = torch.cat(
        (
            initial_seed,
            torch.zeros(1, cfg.max_particle_input - 1, 5, device=cfg.device_acc),
        ),
        dim=1,
    )

    for event in range(nb_events):
        # Run the encoder on the hits
        seed_class, nb_seeds, seed_momentum = model(
            input_tensor_hits[event].unsqueeze(0),
            initial_seed,
            mask_hits,
            mask_particles,
            padding_mask_hit[event].unsqueeze(0),
            None,
        )

        print("testing event ", nb_events + 1)
        print(
            "The number of track is :", len(particles[particles["event_id"] == event])
        )
        print("The number of seed found is ", nb_seeds[0])
        # Compare the seed class and the particle class, for each hit print the bin nb of particle_class different from 0 and the bin nb of the largest bin of seed_class
        for hit in range(seed_class.size(1)):
            particle_bin = particle_class[0, hit]
            seed_bin = torch.argmax(seed_class[0, hit])
            print("Hit", hit)
            print("Particle class bin:", particle_bin.item())
            print("Seed class bin:", seed_bin.item())

        # Print loop over the seed, print them and print the corresponding particle
        seed_i = 0
        for seed in range(seed_momentum.size(1)):
            # Compare the particle and the seed
            print("Seed", seed_i)
            print("Particle", input_tensor_particles[0, seed])
            print("Seed vertex Z", seed_momentum[0, seed][0])
            print("Seed momentum", seed_momentum[0, seed][1:])
            seed_i += 1


def main():
    """
    Main function to run the training of the transformer model for seed reconstruction
    """
    # Parse the command line argument
    cfg = config()
    cfg.parse_args()

    # Set the device to use
    if torch.cuda.is_available():
        cfg.device_acc = torch.device("cuda:0")

    # Print starting information
    print("Starting the training of the transformer model for seed reconstruction")
    cfg.print_config()
    print("Using device:", cfg.device_acc)

    emb = 512

    if cfg.embedding == "ID":
        embedding_encoder = EmbeddingGeoID(
            emb_size=emb,
            max_volume=100,
            max_layer=100,
            max_sensitive=10000,
            max_extra=100,
            dropout=0.0,
            device=cfg.device_acc,
        )
    elif cfg.embedding == "Position":
        embedding_encoder = EmbeddingHitPosition(
            emb_size=emb,
            range_x=[-200, 200],
            range_y=[-200, 200],
            range_z=[-3000, 3000],
            bins_x=100,
            bins_y=100,
            bins_z=100,
            dropout=0.0,
            device=cfg.device_acc,
        )
    elif cfg.embedding == "ID+Position":
        embedding_encoder = EmbeddingHitIDPosition(
            emb_size=emb,
            max_volume=100,
            max_layer=100,
            range_x=[-200, 200],
            range_y=[-200, 200],
            range_z=[-3000, 3000],
            bins_x=100,
            bins_y=100,
            bins_z=100,
            dropout=0.0,
            device=cfg.device_acc,
        )
    else:
        raise ValueError("Embedding mode not recognised")

    # Create the transformer model
    model = SeedTransformer(
        3,
        3,
        emb,
        cfg.max_hit_input,
        4,
        cfg.device_acc,
        embedding_encoder,
        0.1,
        cfg.max_particle_input,
    )
    model.to(cfg.device_acc)
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)

    # # For debugging purpose, print the model parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    if cfg.input_type == "csv":

        # Open the hits and particles csv files
        hits_train, particles_train, nb_events_train = read_data(
            "train/hits.csv", "train/particles.csv", 0, cfg.vertex_cuts
        )
        # hits_train = pd.DataFrame()
        # particles_train = pd.DataFrame()
        # val_fraction = 0.1
        # dir_path = "ODD_data_mu"
        # nb_files = len([name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))])
        # nb_events = 0
        # for i in range(math.floor(nb_files*(1-val_fraction))):
        #     hits, particles, nb_events = read_data("train/odd_full_chain_" + str(i) + "/hits.csv", "train/odd_full_chain_" + str(i) + "/particles.csv", nb_events, cfg.vertex_cuts)
        #     hits_train = pd.concat([hits_train, hits])
        #     particles_train = pd.concat([particles_train, particles])

        (
            input_tensor_hits_train,
            input_tensor_particles_train,
            particle_class_train,
            padding_mask_hit_train,
            padding_mask_particle_train,
        ) = prepare_input_tensor(
            hits_train, particles_train, nb_events_train, cfg, model.embedding_encoder
        )
        # Save the new tensor so we can reuse them for later training
        torch.save(input_tensor_hits_train, "train/input_tensor_hits_train.pt")
        torch.save(
            input_tensor_particles_train, "train/input_tensor_particles_train.pt"
        )
        torch.save(particle_class_train, "train/particle_class_train.pt")
        torch.save(padding_mask_hit_train, "train/padding_mask_hit_train.pt")
        torch.save(padding_mask_particle_train, "train/padding_mask_particle_train.pt")

        hits_val, particles_val, nb_events_val = read_data(
            "val/hits.csv", "val/particles.csv", 0, cfg.vertex_cuts
        )
        # hits_train = pd.DataFrame()
        # particles_train = pd.DataFrame()
        # nb_events = 0
        # for i in range(math.floor(nb_files * (1 - val_fraction)), nb_files):
        #     hits, particles, nb_events = read_data(
        #         "train/odd_full_chain_" + str(i) + "/hits.csv",
        #         "train/odd_full_chain_" + str(i) + "/particles.csv",
        #         nb_events,
        #         cfg.vertex_cuts,
        #     )
        #     hits_train = pd.concat([hits_train, hits])
        #     particles_train = pd.concat([particles_train, particles])

        # Prepare the input tensor and padding mask
        (
            input_tensor_hits_val,
            input_tensor_particles_val,
            particle_class_val,
            padding_mask_hit_val,
            padding_mask_particle_val,
        ) = prepare_input_tensor(
            hits_val, particles_val, nb_events_val, cfg, model.embedding_encoder
        )
        # Save the new tensor so we can reuse them for later training
        torch.save(input_tensor_hits_val, "val/input_tensor_hits_val.pt")
        torch.save(input_tensor_particles_val, "val/input_tensor_particles_val.pt")
        torch.save(particle_class_val, "val/particle_class_val.pt")
        torch.save(padding_mask_hit_val, "val/padding_mask_hit_val.pt")
        torch.save(padding_mask_particle_val, "val/padding_mask_particle_val.pt")

    elif cfg.input_type == "tensor":
        input_tensor_hits_train = torch.load("train/input_tensor_hits_train.pt")
        input_tensor_particles_train = torch.load(
            "train/input_tensor_particles_train.pt"
        )
        particle_class_train = torch.load("train/particle_class_train.pt")
        padding_mask_hit_train = torch.load("train/padding_mask_hit_train.pt")
        padding_mask_particle_train = torch.load("train/padding_mask_particle_train.pt")

        input_tensor_hits_val = torch.load("val/input_tensor_hits_val.pt")
        input_tensor_particles_val = torch.load("val/input_tensor_particles_val.pt")
        particle_class_val = torch.load("val/particle_class_val.pt")
        padding_mask_hit_val = torch.load("val/padding_mask_hit_val.pt")
        padding_mask_particle_val = torch.load("val/padding_mask_particle_val.pt")

    else:
        print("The input type is not recognised")
        exit()

    # For all tensor shuffle by event
    # Shuffle the training tensor
    perm = torch.randperm(input_tensor_hits_train.size(0))
    input_tensor_hits_train = input_tensor_hits_train[perm]
    input_tensor_particles_train = input_tensor_particles_train[perm]
    particle_class_train = particle_class_train[perm]
    padding_mask_hit_train = padding_mask_hit_train[perm]
    padding_mask_particle_train = padding_mask_particle_train[perm]

    # Shuffle the validation tensor
    perm = torch.randperm(input_tensor_hits_val.size(0))
    input_tensor_hits_val = input_tensor_hits_val[perm]
    input_tensor_particles_val = input_tensor_particles_val[perm]
    particle_class_val = particle_class_val[perm]
    padding_mask_hit_val = padding_mask_hit_val[perm]
    padding_mask_particle_val = padding_mask_particle_val[perm]

    # Initialise the metrics
    metrics_train = metrics(cfg.epoch_nb)
    metrics_val = metrics(cfg.epoch_nb)

    for epoch in range(cfg.epoch_nb):
        print("Epoch: ", epoch)

        # Train the model
        model.train()
        model, metrics_train = run_model(
            epoch,
            cfg,
            input_tensor_hits_train,
            input_tensor_particles_train,
            particle_class_train,
            padding_mask_hit_train,
            padding_mask_particle_train,
            model,
            metrics_train,
            opt,
        )

        # Validate the model
        with torch.no_grad():
            # Perform the validation of the model
            # model.eval()
            _, metrics_val = run_model(
                epoch,
                cfg,
                input_tensor_hits_val,
                input_tensor_particles_val,
                particle_class_val,
                padding_mask_hit_val,
                padding_mask_particle_val,
                model,
                metrics_val,
            )

    # Save the model
    torch.save(model, "transformer.pt")

    # Delete all the variable to free some memory
    del model
    del opt

    # Display plot of the loss of the training and validation as a function of the epoch
    plot_loss(metrics_train.loss, metrics_val.loss, "Loss", cfg.interactive)
    plot_loss(
        metrics_train.loss_class, metrics_val.loss_class, "loss_class", cfg.interactive
    )
    plot_loss(
        metrics_train.loss_momentum,
        metrics_val.loss_momentum,
        "Loss_momentum",
        cfg.interactive,
    )
    plot_loss(
        metrics_train.loss_iter, metrics_val.loss_iter, "Loss_iter", cfg.interactive
    )

    # Perform the validation of the model
    model = torch.load("transformer.pt")
    model.to(cfg.device_acc)

    if cfg.event_test > 0:
        # Perform the testing of the model
        hits_test, particles_test, nb_events = read_data(
            "test/hits.csv", "train/particles.csv", 0, cfg.vertex_cuts
        )
        test_model(cfg, hits_test, particles_test, cfg.event_test, model)


if __name__ == "__main__":
    sys.exit(main())
