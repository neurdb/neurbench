import torch
import torchsort


def pearson(tensor: torch.Tensor):
    """
    Compute pairwise correlation between columns of a 2D tensor.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: A 2D tensor of shape (n_features, n_features) representing pairwise correlations.
    """
    # Subtract the mean of each column
    mean_centered = tensor - tensor.mean(dim=0, keepdim=True)

    # Compute the standard deviation of each column
    std_dev = tensor.std(dim=0, unbiased=True, keepdim=True)

    # Normalize the columns to have mean 0 and std 1
    normalized = mean_centered / std_dev

    # Compute the pairwise correlation matrix
    correlation_matrix = (normalized.T @ normalized) / (tensor.size(0) - 1)

    return correlation_matrix


def spearman(tensor: torch.Tensor):
    """
    Compute pairwise Spearman's rank correlation between columns of a 2D tensor.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: A 2D tensor of shape (n_features, n_features) representing pairwise Spearman's rank correlations.
    """
    rank = torchsort.soft_rank(
        tensor.T, regularization="l2", regularization_strength=0.01
    ).T

    return pearson(rank)


def pearson_rel(x, y):
    # Ensure inputs are of the same shape
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")

    # Center the tensors
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    # Compute covariance
    covariance = torch.mean(x_centered * y_centered)

    # Compute standard deviations
    std_x = torch.std(x)
    std_y = torch.std(y)

    # Compute Pearson correlation coefficient
    correlation = covariance / (std_x * std_y)

    return correlation


def spearman_rel(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()
