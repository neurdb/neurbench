import torch
import torchsort

x = torch.tensor(
    [
        [8, 0, 5, 3, 2, 99, 1, 6, 7, 9],
        [-100, -200, -900, -300, -400, -500, -600, -700, -800, 0],
    ]
)

print(torchsort.soft_sort(x, regularization_strength=1.0))
print(torchsort.soft_sort(x, regularization_strength=0.1))
print(torchsort.soft_rank(x))
