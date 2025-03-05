import pandas
import torch


def compute_pairwise_correlation(tensor):
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


# Example usage
data = torch.randn(100, 8)

correlation = compute_pairwise_correlation(data)
print(correlation)

data_df = pandas.DataFrame(data.numpy())
print(data_df.corr())
