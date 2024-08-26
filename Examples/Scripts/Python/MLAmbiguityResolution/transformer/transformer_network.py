import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.nn import Transformer
from embedding_network import EmbeddingGeoID, EmbeddingHitPosition, PositionalEncoder

from typing import Tuple


class SeedTransformer(nn.Module):
    """
    Transformer network for seed finding and track fitting
    Combine a an encoder and decoder to find the best seeds given a could of points
    Members:
        - transformer: Transformer layer
        - embedding_encoder: Embedding layer for the encoder
        - embedding_decoder: Embedding layer for the decoder
        - pos_encoding_decoder: Positional encoding layer for the decoder
        - classify_seed: Linear layer to classify the seed by particle
        - seed_vertex: Linear layer to extract the vertex position from the decoder output
        - seed_momentum: Linear layer to extract the seed momentum from the decoder output
        - keep_iterating: Linear layer to determine whether to keep iterating or not based on the decoder output
    Args:
        - nb_encoder_layers: Number of encoder layers
        - nb_decoder_layers: Number of decoder layers
        - dim_embedding: Dimension of the embedding  (number of dim for the hit encoding)
        - dim_hits: Maximum number of hits in the input
        - nb_head: Number of head for the multihead attention
        - device_acc: Device to run the model on (cpu or cuda)
        - embedding_mode: Mode for the embedding layer (can either use ID, position or both)
        - dropout: Dropout rate
        - dim_seed: Maximum number of reconstructed seed in the output

    """

    def __init__(
        self,
        nb_encoder_layers: int,
        nb_decoder_layers: int,
        dim_embedding: int,
        dim_hits: int,
        nb_head: int,
        device_acc: str,
        embedding_network,
        dropout: float = 0.0,
        dim_seed: int = 100,
    ):
        super(SeedTransformer, self).__init__()
        # Initialise the Transformer model
        self.transformer = Transformer(
            d_model=dim_embedding,
            nhead=nb_head,
            num_encoder_layers=nb_encoder_layers,
            num_decoder_layers=nb_decoder_layers,
            dim_feedforward=dim_hits,
            dropout=dropout,
            device=device_acc,
            batch_first=True,
        )
        self.embedding_encoder = embedding_network

        # Initialise the embedding for the decoder
        self.embedding_decoder = nn.Linear(5, dim_embedding)

        # Positional encoding for the decoder input
        self.pos_encoding_decoder = PositionalEncoder(
            dim_embedding,
            dim_seed,
            dropout,
            device=device_acc,
        )

        # Linear layer to extract the expected number of seed from the encoded information
        self.classify_seed = nn.Linear(
            dim_embedding,
            dim_seed,
            device=device_acc,
        )
        # Linear layer to extract the seed Z0 and momentum from the decoder output
        self.seed_momentum = nn.Linear(dim_embedding, 4, device=device_acc)
        # Linear layer to determine whether to keep iterating or not based on the decoder output
        self.keep_iterating = nn.Linear(dim_embedding, 1, device=device_acc)

        self.keep_sigmoide = nn.Sigmoid().to(device_acc)
        self.class_softMax = nn.Softmax(dim=2).to(device_acc)
        # # First token as a learnable parameter    <= THING ABOUT THIS AT A LATER POINT !!!
        # self.first_token = nn.Parameter(torch.randn(1, 6))

    # def _init_weights(self, module):
    #     """
    #     Initialise the model weights
    #     """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Initialise weights using a normal distribution
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         # Initialise layer normalization parameters
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def encode(
        self, hits: Tensor, mask: Tensor, padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode the input hit sequence.
        Args:
            - hits (Tensor): Input source sequence.
            - mask (Tensor): Source mask.
            - padding_mask (Tensor): Source padding mask.

        Returns:
            - encoded (Tensor): Encoded memory.
            - classify_seed (Tensor):  Attempt to classify the seed by particle
        """
        # Loop over the entry in the batch and run the embedding layer
        embedded_src = self.embedding_encoder(hits)
        # Encode the source sequence
        encoded = self.transformer.encoder(
            src=embedded_src, mask=mask, src_key_padding_mask=padding_mask
        )
        return encoded, self.class_softMax(self.classify_seed(encoded))

    def decode(
        self,
        seeds: Tensor,
        encoded: Tensor,
        mask: Tensor,
        padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Decode the seed from previously encoded information and reconstructed seeds.

        Args:
            - seeds (Tensor): Output seed sequence.
            - encoded (Tensor): Encoded information of the hits from the encoder.
            - mask (Tensor): Mask on the seed.
            - padding_mask (Tensor): Padding mask for the seed.

        Returns:
            - seed_vertex (Tensor): Vertex position.
            - seed_momentum (Tensor): Seed momentum.
            - keep_sigmoide (Tensor): Whether to keep iterating.
        """
        # Embed the target sequence
        # tgt_emb = self.pos_encoding_decoder(self.embedding_decoder(seeds))
        tgt_emb = self.embedding_decoder(seeds)
        # Decode the target sequence
        reconstructed_seeds = self.transformer.decoder(
            tgt=tgt_emb,
            memory=encoded,
            tgt_mask=mask,
            memory_mask=None,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=None,
        )

        return (
            self.seed_momentum(reconstructed_seeds),
            self.keep_sigmoide(self.keep_iterating(reconstructed_seeds)),
        )

    def forward(
        self,
        hits: Tensor,
        seed: Tensor,
        mask_hits: Tensor,
        mask_seed: Tensor,
        padding_mask_hits: Tensor,
        padding_mask_seed: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the transformer network.
        Args:
            - hits (Tensor): Input source sequence.
            - seed (Tensor): Output seed sequence.
            - mask_hits (Tensor): Source mask.
            - mask_seed (Tensor): Mask on the seed.
            - padding_mask_hits (Tensor): Source padding mask.
            - padding_mask_seed (Tensor): Padding mask for the seed.
        Returns:
            - nb_seeds_encoder (Tensor): Number of seeds found.
            - nb_seeds (Tensor): Number of seeds found.
            - seed_vertex (Tensor): Vertex position.
            - seed_momentum (Tensor): Seed momentum.
        """

        iter_threshold = 0.1
        # Encode the source sequence
        encoded, seed_class = self.encode(hits, mask_hits, padding_mask_hits)
        nb_loop = 0
        keep_iteration = True
        nb_seeds = Tensor(seed.size(0)).to(seed.device)
        seed_momentum = Tensor(seed.size(0), seed.size(1), 4).to(seed.device)

        while nb_loop < mask_hits.size(0) and keep_iteration:
            # Decode the target sequence
            seed_momentum, keep = self.decode(
                seed, encoded, mask_seed, padding_mask_seed
            )
            # Check if all the value of keep at rank nb_loop (for all the batch entry)  are below the threshold
            keep_iteration = torch.all(keep[:, nb_loop] < iter_threshold)

        # For each batch set to 0 all entry after the first time keep is below the threshold
        for batch in range(seed.size(0)):
            for hits in range(seed.size(1)):
                if keep[batch, hits] < iter_threshold:
                    seed_momentum[batch, hits, :] = 0
                    nb_seeds[batch] = hits
                    break

        return seed_class, nb_seeds, seed_momentum
