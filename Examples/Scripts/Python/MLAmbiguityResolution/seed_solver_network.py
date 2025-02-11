import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import ast


def prepareDataSet(data: pd.DataFrame) -> pd.DataFrame:
    """Format the dataset that have been written from the Csv file"""
    """
    @param[in] data: input DataFrame containing 1 event
    @return: Formatted DataFrame 
    """
    # Remove entry with no good seed
    good_matched = data.loc[data["rank"] == 0, "particleId"].unique()
    data = data[data["particleId"].isin(good_matched)]
    # Sort by particle ID
    data = data.sort_values("particleId")
    # Set truth particle ID as index
    data = data.set_index("particleId")
    # Transform the hit list from a string to an actual list
    hitsIds = []
    for list in data["Hits_ID"].values:
        hitsIds.append(ast.literal_eval(list))
    data["Hits_ID"] = hitsIds
    return data


class DuplicateClassifier(nn.Module):
    """MLP model used to separate goods seed from duplicate seeds. Return one score per seed the higher one correspond to the good seed."""

    def __init__(self, input_dim, n_layers, marginDuplicate=0.01):
        """Four layer MLP, sigmoid activation for the last layer."""
        super(DuplicateClassifier, self).__init__()
        if(len(n_layers)==0):
            raise RuntimeError("The networks need at least one layer")
        self.input = nn.Linear(input_dim, n_layers[0])
        self.linear = []
        for i_layer in range(len(n_layers)-1):
            self.linear.append(nn.Linear(n_layers[i_layer], n_layers[i_layer+1]))
        self.linear = nn.ModuleList(self.linear)
        self.output = nn.Linear(n_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("marginDuplicate", torch.tensor(marginDuplicate, dtype=torch.float32))

    def forward(self, z):
        z = F.relu(self.input(z))
        for i_layer in range(len(self.linear)):
            z = self.linear[i_layer](z)
        return self.sigmoid(self.output(z))


class Normalise(nn.Module):
    """Normalisation of the input before the MLP model."""

    def __init__(self, mean, std):
        super(Normalise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, z):
        z = z - self.mean
        z = z / self.std
        return z

    def _apply(self, fn):
        new_self = super(Normalise, self)._apply(fn)
        new_self.mean = fn(new_self.mean)
        new_self.std = fn(new_self.std)
        return new_self

