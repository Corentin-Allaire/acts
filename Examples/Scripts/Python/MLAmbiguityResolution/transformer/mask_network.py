import torch


def build_look_ahead_mask(size: int, device: torch.device):
    """
    Function that build a diagonal mask for the transformer model
    Used to ignore further tokens in the sequence (both in the encoder and decoder?)
    Args:
        - size (int): Size of the mask
        - device (torch.device): Device to use
    Returns:
        - torch.Tensor: The look ahead mask
    """
    square = torch.ones(size, size)
    mask = square.triu(diagonal=1)
    mask = mask * float("-inf")
    mask[torch.isnan(mask)] = 1
    mask.to(device)
    return mask


def build_mask_volume_layer(data: torch.Tensor, device: torch.device):
    """
    Build a mask that will ignore all the input with the same volume ID and lower and equal layer ID
    Args:
        - data (torch.Tensor): The input data
        - device (torch.device): Device to use
    Returns:
        - torch.Tensor: The mask
    """
    # Extract the volume and layer ID
    volume = data["volume"]
    layer = data["layer"]
    # Build the mask
    mask = torch.zeros(data.size(0), data.size(0))
    for i in range(data.size(0)):
        for j in range(data.size(0)):
            mask[i, j] = (volume[j] == volume[i]) & (layer[j] <= layer[i])
    mask = mask * float("-inf")
    mask[torch.isnan(mask)] = 1
    mask.to(device)
    return mask
