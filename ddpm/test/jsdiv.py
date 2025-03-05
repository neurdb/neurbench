import torch
import torch.nn.functional as F


def js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence for each pair of distributions in the batch.
    p, q: 2-D PyTorch tensors with shape (batch_size, num_classes).
    Each row must sum to 1 (representing a valid probability distribution).
    Returns: A 1-D tensor of JS divergence values for each pair.
    """
    # Compute the midpoint distribution for each pair in the batch
    m = 0.5 * (p + q)

    # Compute KL divergence for each pair (row-wise)
    kl_pm = F.kl_div(p.log(), m, reduction="none").sum(dim=1)  # Sum over classes
    kl_qm = F.kl_div(q.log(), m, reduction="none").sum(dim=1)  # Sum over classes

    # Compute JS divergence for each pair
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div


# Example: Batch of probability distributions
p = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.5, 0.3]], dtype=torch.float32)
q = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]], dtype=torch.float32)

# Normalize to ensure valid probability distributions
p = p / p.sum(dim=1, keepdim=True)
q = q / q.sum(dim=1, keepdim=True)

# Compute JS divergence for each pair using PyTorch
js_div_pytorch = js_divergence(p, q)
print(f"JS Divergence (PyTorch): {js_div_pytorch}")

# For comparison: Compute JS divergence for each pair using Scipy
from scipy.spatial.distance import jensenshannon

p_np = p.numpy()
q_np = q.numpy()
js_div_scipy = torch.tensor(
    [jensenshannon(p_np[i], q_np[i]) ** 2 for i in range(len(p_np))],
    dtype=torch.float32,
)
print(f"JS Divergence (Scipy): {js_div_scipy}")

# Verify the results
print(
    f"Difference between PyTorch and Scipy: {torch.abs(js_div_pytorch - js_div_scipy)}"
)
