import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import math
from torch import Tensor


def sort_by_position(data: pd.DataFrame) -> pd.DataFrame:
    """Sort the hits by their radial position the by z position"" if they are at the same radial position
    Args:
        - data: input DataFrame containing 1 event
    Returns:
        - Sorted DataFrame
    """
    data = data.sort_values(by=["r", "z"], key=abs)
    return data    

class EmbeddingGeoID(nn.Module):
    """
    Embedding class for the geometry ID of the hit
    This class  embed the geometry id of the hit into a higher dimensional space
    Args:
        - emb_size (int): Dimension of the embedding
        - max_volume (int): Maximum number of volume
        - max_layer (int): Maximum number of layer
        - max_sensitive (int): Maximum number of sensitive volume
        - max_extra (int): Maximum number of extra information
        - dropout (float): Dropout rate
        - device (str): Device to run the model on (cpu or cuda
    Members:
        - embeddingVolume: Embedding layer for the volume
        - embeddingLayer: Embedding layer for the layer
        - embeddingSensitive: Embedding layer for the sensitive volume
        - embeddingExtra: Embedding layer for the extra information
        - dropout: Dropout layer
    """
    def __init__(self, emb_size: int, max_volume: int, max_layer: int, max_sensitive: int, max_extra: int, dropout: float = 0.1, device: str = "cpu"):
        super(EmbeddingGeoID, self).__init__()
        # Embedding layers
        self.embeddingVolume = nn.Embedding(max_volume, emb_size, padding_idx=0, device=device)
        self.embeddingLayer = nn.Embedding(max_layer, emb_size, padding_idx=0, device=device)
        self.embeddingSensitive = nn.Embedding(max_sensitive, emb_size, padding_idx=0, device=device)
        self.embeddingExtra = nn.Embedding(
            max_extra, emb_size, padding_idx=0, device=device
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed the volume, layer, sensitive and extra values
        Args:
            - x (Tensor): Input tensor containing the volume, layer, sensitive and extra values
        Returns:
            - Tensor: The embedded values
        """
        volume = self.embeddingVolume(x[:,:,0].int())
        layer = self.embeddingLayer(x[:, :, 1].int())
        sensitive = self.embeddingSensitive(x[:, :, 2].int())
        extra = self.embeddingExtra(x[:, :, 3].int())
        # Sum the embeddings
        sum_embedding = volume + layer + sensitive + extra
        return self.dropout(sum_embedding)

# Define an embedding class for the encoder input
# This class will embed the hit position into a higher dimensional space
# To achieve this the position will first need to be binned
class EmbeddingHitPosition(nn.Module):
    """
    Embedding based on the position of the hits
    This class embed the hit position into a higher dimensional space
    To achieve this the position will first need to be binned
    Args:
        - emb_size (int): Dimension of the embedding
        - range_x (list): Range of the x position
        - range_y (list): Range of the y position
        - range_z (list): Range of the z position
        - bins_x (int): Number of bins for the x position
        - bins_y (int): Number of bins for the y position
        - bins_z (int): Number of bins for the z position
        - dropout (float): Dropout rate
        - device (str): Device to run the model on (cpu or cuda)  
    Members:
        - range_x: Range of the x position
        - range_y: Range of the y position
        - range_z: Range of the z position
        - bins_x: Number of bins for the x position
        - bins_y: Number of bins for the y position
        - bins_z: Number of bins for the z position
        - embedding_x: Embedding layer for the x position
        - embedding_y: Embedding layer for the y position
        - embedding_z: Embedding layer for the z position
        - dropout: Dropout layer
    """
    def __init__(
        self,
        emb_size: int,
        range_x: list = [-3000, 3000],
        range_y: list = [-3000, 3000],
        range_z: list = [-3000, 3000],
        bins_x: int = 100,
        bins_y: int = 100,
        bins_z: int = 100,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super(EmbeddingHitPosition, self).__init__()
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.bins_x = bins_x
        self.bins_y = bins_y
        self.bins_z = bins_z
        # Embedding layer !!!! NEED TO THINK OF THE PADDING !!!!
        self.embedding_x = nn.Embedding(bins_x, emb_size, padding_idx=0, device=device)
        self.embedding_y = nn.Embedding(bins_y, emb_size, padding_idx=0, device=device)
        self.embedding_z = nn.Embedding(bins_z, emb_size, padding_idx=0, device=device)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed the hit position
        Args:
            - x (Tensor): Input tensor containing the hit position
        Returns:
            - Tensor: The embedded hit position
        """
        # Bin the hit position
        x_bin = torch.ceil(
            (x[:, :, 0] - self.range_x[0])
            / (self.range_x[1] - self.range_x[0])
            * self.bins_x
        )
        y_bin = torch.ceil(
            (x[:, :, 1] - self.range_y[0])
            / (self.range_y[1] - self.range_y[0])
            * self.bins_y
        )
        z_bin = torch.ceil(
            (x[:, :, 2] - self.range_z[0])
            / (self.range_z[1] - self.range_z[0])
            * self.bins_z
        )
        # Embed the hit position
        x_emb = self.embedding_x(x_bin)
        y_emb = self.embedding_y(y_bin)
        z_emb = self.embedding_z(z_bin)
        # Sum the embeddings
        sum_embedding = x_emb + y_emb + z_emb
        return self.dropout(sum_embedding)

class EmbeddingHitIDPosition(nn.Module):
    """
    Embedding based on the position of the hits and the volume and layer ID
    This class embed the hit information into a higher dimensional space
    To achieve this the position will first need to be binned
    Args:
        - emb_size (int): Dimension of the embedding
        - range_x (list): Range of the x position
        - range_y (list): Range of the y position
        - range_z (list): Range of the z position
        - bins_x (int): Number of bins for the x position
        - bins_y (int): Number of bins for the y position
        - bins_z (int): Number of bins for the z position
        - dropout (float): Dropout rate
        - device (str): Device to run the model on (cpu or cuda) 
    Members:
        - range_x: Range of the x position
        - range_y: Range of the y position
        - range_z: Range of the z position
        - bins_x: Number of bins for the x position
        - bins_y: Number of bins for the y position
        - bins_z: Number of bins for the z position
        - embeddingVolume: Embedding layer for the volume
        - embeddingLayer: Embedding layer for the layer
        - embedding_x: Embedding layer for the x position
        - embedding_y: Embedding layer for the y position
        - embedding_z: Embedding layer for the z position
        - dropout: Dropout layer 
    """
    def __init__(
        self,
        emb_size: int,
        max_volume: int,
        max_layer: int,
        range_x: list = [-3000, 3000],
        range_y: list = [-3000, 3000],
        range_z: list = [-3000, 3000],
        bins_x: int = 100,
        bins_y: int = 100,
        bins_z: int = 100,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super(EmbeddingHitPosition, self).__init__()
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.bins_x = bins_x
        self.bins_y = bins_y
        self.bins_z = bins_z
        # Embedding layer !!!! NEED TO THINK OF THE PADDING !!!!
        self.embeddingVolume = nn.Embedding(max_volume, emb_size, padding_idx=0, device=device)
        self.embeddingLayer = nn.Embedding(max_layer, emb_size, padding_idx=0, device=device)
        self.embedding_x = nn.Embedding(bins_x, emb_size, padding_idx=0, device=device)
        self.embedding_y = nn.Embedding(bins_y, emb_size, padding_idx=0, device=device)
        self.embedding_z = nn.Embedding(bins_z, emb_size, padding_idx=0, device=device)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Bin the hit position
        x_bin = torch.ceil(
            (x[:, :, 2] - self.range_x[0])
            / (self.range_x[1] - self.range_x[0])
            * self.bins_x
        )
        y_bin = torch.ceil(
            (x[:, :, 3] - self.range_y[0])
            / (self.range_y[1] - self.range_y[0])
            * self.bins_y
        )
        z_bin = torch.ceil(
            (x[:, :, 4] - self.range_z[0])
            / (self.range_z[1] - self.range_z[0])
            * self.bins_z
        )

        volume = self.embeddingVolume(x[:, :, 0])
        layer = self.embeddingLayer(x[:, :, 1])

        # Embed the hit position
        x_emb = self.embedding_x(x_bin)
        y_emb = self.embedding_y(y_bin)
        z_emb = self.embedding_z(z_bin)
        # Sum the embeddings
        sum_embedding = volume + layer + x_emb + y_emb + z_emb
        return self.dropout(sum_embedding)


# Define a positional encoding class for the decoder input
class PositionalEncoder(nn.Module):
    def __init__(self, emb_size: int, max_seq_len: int, dropout: float = 0.1, device: str = "cpu"):
        super(PositionalEncoder, self).__init__()
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout)
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, emb_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe.to(device)
        position.to(device)
        div_term.to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.emb_size)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
